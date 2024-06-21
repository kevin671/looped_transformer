"""
class LoopedModel:
    def forward():
        for i in self.n_loop:
            x = self.f(x)
        return x

Now we wanna make faster by parallelizing the loop. 

x_target = [f(x) for i in range(n_loop)]

this is initial value of x_target. 
Then we can update this value by using picard iteration.
xs = [x for i in range(n_loop)]


batch_window_size = 4
start = 0
for:
    xs = xs[start:start+batch_window_size]
    drift = compute_drift(xs)
    update_value(xs, drift)
    s = slide_window(xs, xs_new)
    start += s

def compute_drift(xs):
    return x

def update_value(xs, drift):
    return x

def slide_window(xs, xs_new):
    threshold = 1e-3
    slide_window_size = f(np.linalg.norm(xs_new - xs), threshold)
    return slide_window_size

"""