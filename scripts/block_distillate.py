import os
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from curriculum import Curriculum
from schema import schema
from models import build_model
from tasks import get_task_sampler
from main_utils import init_device, get_run_id, load_pretrained_model

# from eval import get_run_metrics


import wandb

torch.backends.cudnn.benchmark = True


def calculate_gradient_norm(model):
    total_norm = 0.0
    norm_dict = {}
    for n, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        norm_dict[n] = param_norm
    total_norm = total_norm ** (1.0 / 2)
    return norm_dict, total_norm


def train_step(
    args,
    teacher_n_loops,
    student_n_loops,
    teacher_model,
    student_model,
    xs,
    ys,
    optimizer,
    ctx,
    scaler,
):
    if ctx is not None:
        raise NotImplementedError

    teacher_model.eval()
    with torch.no_grad():
        # _, teacher_output_list = teacher_model(xs, ys, 0, teacher_n_loops, return_output=True)
        # list of [B, 2n, n_embd], length = teacher_n_loops
        teacher_y_pred = teacher_model(xs, ys, 0, teacher_n_loops)

    student_y_pred = student_model(xs, ys, 0, student_n_loops)
    n_loop_ratio = teacher_n_loops // student_n_loops

    total_loss = 0
    loss = 0
    student_model.train()
    dist_loops = 1
    for i in range(dist_loops):
        loss += (
            (student_y_pred[i] - teacher_y_pred[(i + 1) * n_loop_ratio - 1])
            .square()
            .mean()
        )

    if args.training.use_ctx:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    total_loss = loss / dist_loops
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        y_pred = student_model(xs, ys, 0, student_n_loops)[-1]
    # norm_dict, total_norm = calculate_gradient_norm(student_model)
    total_norm = 0
    norm_dict = {}
    return total_loss.detach(), y_pred.detach(), total_norm, norm_dict


def main(args, device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    dtype = "float16"  # 'bfloat16', 'float32'
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    if args.training.use_ctx:
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype, cache_enabled=False)
    else:
        ctx = None
    ################################################
    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        mode="disabled" if args.debug_mode else "online",
        resume=True,
    )

    torch.manual_seed(args.training.seed)

    teacher_model = build_model(args.model)
    teacher_model.to(device)
    teacher_model.eval()

    student_model = build_model(args.model)
    student_model.to(device)
    student_model.train()

    teacher_n_loops = args.progressive_distillation.teacher_n_loops
    student_n_loops = args.progressive_distillation.student_n_loops

    assert teacher_n_loops % student_n_loops == 0

    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=args.training.learning_rate,
        weight_decay=args.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    curriculum = Curriculum(args.training.curriculum)

    # Here the model load the pretrained teacher model
    args, teacher_model, optimizer, _, state_path, _ = load_pretrained_model(
        args, teacher_model, optimizer, curriculum, device
    )

    if args.training.use_fixed_dataset:
        from main_utils import gen_dataloader

        task_sampler = get_task_sampler(
            task_name=args.training.task_name,
            batch_size=args.training.batch_size,
            n_points=args.training.curriculum.points.end,
            n_dims=args.model.n_dims,
            n_dims_truncated=args.training.curriculum.dims.end,  # == args.model.n_dimss
            device=device,
            sparsity=args.training.sparsity,
        )
        train_loader = gen_dataloader(
            task_sampler, args.training.train_size, args.training.batch_size
        )
        train_iter = iter(train_loader)
        test_loader = gen_dataloader(
            task_sampler, args.training.test_size, args.training.batch_size
        )

    pbar = tqdm(range(args.training.train_steps))
    for i in pbar:
        if args.training.use_fixed_dataset:
            try:
                batch = next(train_iter)
                xs, ys = batch["x"].to(device), batch["y"].to(device)
            except StopIteration:
                train_iter = iter(train_loader)
        else:
            task_sampler = get_task_sampler(
                task_name=args.training.task_name,
                batch_size=args.training.batch_size,
                n_points=args.training.curriculum.points.end,
                n_dims=args.model.n_dims,
                n_dims_truncated=args.training.curriculum.dims.end,
                device=device,
                sparsity=args.training.sparsity,
            )

            real_task = task_sampler()
            xs, ys = real_task.xs.float(), real_task.ys.float()

        loss, output, total_norm, grad_norm_dict = train_step(
            args,
            teacher_n_loops,
            student_n_loops,
            teacher_model,
            student_model,
            xs,
            ys,
            optimizer,
            ctx,
            scaler,
        )

        # EVALUATION ======================================
        point_wise_tags = list(
            range(args.training.curriculum.points.end)
        )  # [0, 1, 2, ..., n-1]
        if i % args.wandb.log_every_steps == 0:
            point_wise_loss = (output - ys).square().mean(dim=0)  # [n,]
            if args.training.use_fixed_dataset:
                # eval
                with torch.no_grad():
                    for batch in test_loader:
                        xs, ys = batch["x"].to(device), batch["y"].to(device)
                        if args.model.family in ["gpt2"]:
                            output = model(xs, ys)  # [B,]
                        elif args.model.family in ["gpt2_loop"]:
                            # student_n_loops = args.training.curriculum.loops.end // 2  # student
                            student_n_loops = 1
                            y_pred_list = student_model(xs, ys, 0, student_n_loops)
                            output = y_pred_list[-1]  # [B, n]
                        else:
                            raise NotImplementedError
                        point_wise_loss = (output - ys).square().mean(dim=0)
                        loss = point_wise_loss.mean()
            wandb.log(
                {
                    "overall_loss": loss,
                    "loop_times": args.training.curriculum.loops.end,
                    # "grad_norm/layerwise": grad_norm_dict,
                    # "grad_norm": total_norm,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
                    ),
                    "n_points": args.training.curriculum.points.end,
                    "n_dims": args.training.curriculum.dims.end,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=i,
            )

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0:
            training_state = {
                "model_state_dict": student_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and i > 0
        ) or (i == args.training.train_steps - 1):
            torch.save(
                {"model": student_model.state_dict()},
                os.path.join(args.out_dir, f"model_{i}.pt"),
            )


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    run_id = args.training.resume_id
    if run_id is None:
        run_id = get_run_id(args)

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    # add a timestamp here, if resumed, this will be the resumed time
    args.wandb["timestamp"] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, device)
