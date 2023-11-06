import numpy
import pickle


CL_TO_INT = {
   "gaussian": 0,
   "skew_gaussian": 1,
   "lorentzian": 2,
}
INT_TO_CL = {
   0: "gaussian",
   1: "skew_gaussian",
   2: "lorentzian",
}

RANDOM_SIGMA = 0.005

def gaussian(x, A, mu, sig, nsigma=4, randomness=True):
    onset = max(mu - nsigma * sig, min(x))
    end = min(mu + nsigma * sig, max(x))
    curve = (A * numpy.exp(-(x-mu)**2/sig**2)
             + randomness * numpy.random.normal(0, RANDOM_SIGMA, len(x)))
    return (curve, onset, end)

def skew_gaussian(x, A, mu, sig, skewness, nsigma=4, randomness=True):
    onset = max(mu - nsigma * sig, min(x))
    end = min(mu + nsigma * sig, max(x))
    gaussian_curve, o, e = gaussian(x, 1, mu, sig, randomness=False)
    curve = (A / 2 * (1 + numpy.tanh((x-mu)*skewness)) * gaussian_curve
             + randomness * numpy.random.normal(0, RANDOM_SIGMA, len(x)))
    return (curve, onset, end)

def lorentzian(x, A, mu, gamma, randomness=True):
    onset = max(mu - 2.5 * gamma, min(x))
    end = min(mu + 2.5 * gamma, max(x))
    curve = (A * (gamma**2 / (gamma**2 + (x-mu)**2))
             + randomness * numpy.random.normal(0, RANDOM_SIGMA, len(x)))
    return (curve, onset, end)

def null_curve(x, randomness=True):
    return (numpy.zeros(len(x))
            + randomness * numpy.random.normal(0, RANDOM_SIGMA, len(x)))

if __name__ == "__main__":
    NX = 416

    XMIN = 0
    XMAX = 50
    DX = XMAX - XMIN
    x = numpy.linspace(XMIN, XMAX, NX)
    print("generating curves...")

    N_curves = 10000
    random_sigmas = numpy.random.uniform(1, 4, N_curves)
    random_mus = numpy.random.uniform(20, 40, N_curves)
    random_amplitudes = numpy.random.uniform(1, 3, N_curves)

    curves = []
    for A, mu, sig in zip(random_amplitudes, random_mus, random_sigmas):
        gaussian_, onset, end = gaussian(x, A, mu, sig,
                                         randomness=False)
        target_label = (onset, end, "gaussian")
        curves.append((gaussian_, [target_label]))

    random_sigmas = numpy.random.uniform(1, 4, N_curves)
    random_mus = numpy.random.uniform(20, 40, N_curves)
    random_amplitudes = numpy.random.uniform(1, 3, N_curves)
    random_skewness = numpy.random.uniform(-2, -1, N_curves)

    for A, mu, sig, skew in zip(random_skewness, random_mus, random_sigmas, random_skewness):
        skew_gaussian_, onset, end = skew_gaussian(x, A, mu, sig, skew,
                                                   randomness=False)
        target_label = (onset, end, "skew_gaussian")
        curves.append((skew_gaussian_, [target_label]))

    random_sigmas = numpy.random.uniform(1, 4, N_curves)
    random_mus = numpy.random.uniform(20, 40, N_curves)
    random_amplitudes = numpy.random.uniform(1, 3, N_curves)
    random_skewness = numpy.random.uniform(1, 2, N_curves)

    for A, mu, sig, skew in zip(random_amplitudes, random_mus, random_sigmas, random_skewness):
        skew_gaussian_, onset, end = skew_gaussian(x, A, mu, sig, skew,
                                                   randomness=False)
        target_label = (onset, end, "skew_gaussian")
        curves.append((skew_gaussian_, [target_label]))

    random_gammas = numpy.random.uniform(1, 2, N_curves)
    random_mus = numpy.random.uniform(20, 40, N_curves)
    random_amplitudes = numpy.random.uniform(1, 3, N_curves)

    for A, mu, gamma in zip(random_amplitudes, random_mus, random_gammas):
        lorentzian_, onset, end = lorentzian(x, A, mu, gamma, randomness=False)
        target_label = (onset, end, "lorentzian")
        curves.append((lorentzian_, [target_label]))

    # TODO generate curves with mixed data (i.e gaussian + lorentzian)

    # for _ in range(N_curves):
    #     null_curve_ = null_curve(x)
    #     curves.append((null_curve_, "null"))

    # print("sorting curves and labels...")
    # X = numpy.array([c[0] for c in curves])
    # y = numpy.array([c[1] for c in curves])
    # data = (X, y)
    # with open("curves.pkl", "wb") as fhandle:
    #     pickle.dump(data, fhandle)

    # YOLOv3-1D format saving
    import os
    #os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "1d_series"), exist_ok=True)
    os.makedirs(os.path.join("data", "labels"), exist_ok=True)
    curves_series = numpy.array([c[0] for c in curves])
    if True:
        print("normalizing curves...")
        abs_max = numpy.max(numpy.abs(curves_series))
        curves_series_normalized = curves_series/abs_max
        print("saving curves...")
        for i, s in enumerate(curves_series_normalized):
            numpy.save(os.path.join("data", "test_data", "1d_series", f"series_{i}.npy"), s)
    # unpack targets
    curves_labels = [c[1] for c in curves]
    # convert string labels to integers
    print("preparing labels...")
    plot_curves = False
    if True:
        import matplotlib.pyplot as plt
        for i, targets in enumerate(curves_labels):
            for j in range(len(targets)):
                # inplace conversion of string label to integer
                onset, end, cl = targets[j]
                mean = (end + onset)/2
                width = (end - onset)
                mean_norm = (mean - XMIN)/DX
                width_norm = (width)/DX
                targets[j] = (mean_norm, width_norm, CL_TO_INT[cl])
            if plot_curves:
                plt.plot(x/DX, curves_series_normalized[i])
                x0 = mean_norm - width_norm/2
                x1 = mean_norm + width_norm/2
                plt.axvspan(x0, x1, alpha=0.5, color="green")
                plt.show()
    print("saving labels...")
    if True:
        # save tragets as csv files
        for i, t in enumerate(curves_labels):
            numpy.savetxt(os.path.join("data", "test_data", "labels", f"labels_{i}.csv"),
                        t, delimiter=" ")
    # save annotations file
    print("saving annotiations file...")
    annotations = [(f"series_{i}.npy",
                    f"labels_{i}.csv")
                   for i, _ in enumerate(curves_labels)]
    numpy.savetxt(os.path.join("data", "test_data", "annotations.csv"), annotations,
                  delimiter=", ", fmt="%s")

    PERCENTAGE_TRAIN = 0.95
    N_CUTOFF = int(len(annotations)*0.95)

    shuffeled_annotations = numpy.random.shuffle(annotations)

    numpy.savetxt(os.path.join("data", "test_data", "train_annotations.csv"),
                  shuffeled_annotations[:N_CUTOFF],
                  delimiter=", ", fmt="%s")
    numpy.savetxt(os.path.join("data", "test_data", "test_annotations.csv"),
                  shuffeled_annotations[N_CUTOFF:],
                  delimiter=", ", fmt="%s")

