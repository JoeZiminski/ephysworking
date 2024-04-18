import matplotlib.pyplot as plt
import numpy as np

# TODO: this is super slow brute force optimisation approach
# but the histograms are so small it doesn't matter.

# TODO:
# next try a multivariate optimisation to find the center point
# calculate O(n)

#  GENERAL UTILS


def shift(trace, shift):
    # can just use np.roll... will still need to zero pad etc.
    TRACE = np.fft.fft(trace)
    f = np.arange(TRACE.size) * (1 / TRACE.size)
    TRACE_SHIFT = np.exp(-1j * 2 * np.pi * f * shift) * TRACE
    trace_shift = np.real(np.fft.ifft(TRACE_SHIFT))
    return trace_shift


def loss(d2, *args):
    (
        t1,
        t2,
    ) = args
    t2_shift = shift(t2, d2)
    return np.sum(
        (t1 - t2_shift) ** 2
    )  # -np.dot(t1, t2_shift) # -np.correlate(t1, t2_shift)  # np.sum((t1-t2_shift)**2)


def calculate_shift(hist_1, hist_2):
    """
    For now a super-naive versiont that performs integer
    shifts and computers brute-force loss over all shifts.
    As these histograms are < 1000 samples it is not an issue.

    It would be nicer to perform real-value shifts and
    optimise with proper algorithm, seems fiddly to restrict
    optimisation to integers with most algorithms. However
    non-integer Fourier shifts are a bit fiddly
    https://stackoverflow.com/questions/21830878/shift-the-elements-of-a-vector-by-non-integer-shift-in-matlab
    and probably not worth it. Alternatively interpolate but I think
    would be slower for accurate interpolations.
    """
    assert hist_1.size == hist_2.size

    N = hist_1.size
    hist_1 = np.r_[np.zeros(N), hist_1, np.zeros(N)]
    hist_2 = np.r_[np.zeros(N), hist_2, np.zeros(N)]

    # TODO: need to account for non-zero in section only and avg.
    shift_range = int(N / 2)
    range_ = np.arange(-shift_range, shift_range)
    loss_ = [loss(d2, hist_1, hist_2) for d2 in range_]
    return range_[np.argmin(loss_)]


def plot_loss(hist_1, hist_2):
    """
    see calculate_shift()
    """
    N = hist_1.size
    hist_1 = np.r_[np.zeros(N), t1, np.zeros(N)]
    hist_2 = np.r_[np.zeros(N), t2, np.zeros(N)]

    shift_range = int(hist_1.size / 2)
    range_ = np.arange(-shift_range, shift_range)

    [plt.plot(shift(hist_2, d2)) for d2 in range_]
    plt.show()

    loss_ = [loss(d2, hist_1, hist_2) for d2 in range_]
    plt.plot(range_, loss_)
    plt.show()


if __name__ == "__main__":

    np.random.seed(41)

    t1 = np.random.randn(300)
    t1[50] *= 100
    t2 = np.r_[np.random.randn(10), t1[:-10]]

    plt.plot(t1)
    plt.plot(t2)

    out = calculate_shift(t1, t2)

    plt.plot(shift(t2, out))
    plt.show()

    plot_loss(t1, t2)
