import abc
import numpy
import torch


class Augmentation(dict):

    def __init__(self, series, bboxes, *args, **kwargs):
        super().__init__(*args, series=series, bboxes=bboxes, **kwargs)


class BaseDistribution(abc.ABC):

    def __init__(self, deterministic=False):
        self._deterministic = deterministic

    def __repr__(self):
        return self.__str__()

    def gen_value(self, deterministic=None):
        deterministic = self._deterministic if deterministic is None else deterministic
        if deterministic:
            return self._deterministic_strategy()
        else:
            return self._indeterministic_strategy()

    @abc.abstractmethod
    def _deterministic_strategy(self):
        pass

    @abc.abstractmethod
    def _indeterministic_strategy(self):
        pass


class Constant(BaseDistribution):

    def __init__(self, value):
        super().__init__(deterministic=True)
        self.__value = value

    def __str__(self):
        return f"Constant(value={self.__value})"

    def _deterministic_strategy(self):
        return self.__value

    def _indeterministic_strategy(self):
        raise NotImplementedError


class NormalDistribution(BaseDistribution):

    def __init__(self, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mean = mean
        self.__std = std

    def __str__(self):
        return f"NormalDistribution(mean={self.__mean}, std={self.__std})"

    def _indeterministic_strategy(self):
        return numpy.random.normal(loc=self.__mean, scale=self.__std)

    def _deterministic_strategy(self):
        return self.__mean


class UniformDistribution(BaseDistribution):

    def __init__(self, low, high, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__low = low
        self.__high = high

    def __str__(self):
        return f"UniformDistribution(low={self.__low}, high={self.__high})"

    def _indeterministic_strategy(self):
        return numpy.random.uniform(low=self.__low, high=self.__high)

    def _deterministic_strategy(self):
        return (self.low + self.high)/2


class BaseTransform(abc.ABC):

    def __init__(self, p, deterministic=False):
        self._p = p
        self._deterministic = deterministic
        self._cache = dict()

    def __call__(self, series, bboxes) -> Augmentation:
        if numpy.random.random() <= self._p:
            return self.transform(series, bboxes)
        else:
            return Augmentation(series, bboxes)

    @abc.abstractmethod
    def transform(self, series, bboxes) -> Augmentation:
        pass

    def toggle_deterministic(self):
        self._deterministic = not self._deterministic


class ComposedTransform(BaseTransform):
    """Compose multiple transforms after one another"""

    def __init__(self, transforms: list[BaseTransform], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def transform(self, series, bboxes):
        for i, transform in enumerate(self.transforms):
            augmentation = transform(series, bboxes)
            series = augmentation.get("series", None)
            bboxes = augmentation.get("bboxes", None)
            self._cache.update({
                f"{i}_{type(transform).__name__}": {
                    "cache": transform._cache,
                    "augmentation": augmentation},
            })
        return Augmentation(series, bboxes)


class PolynomialBaseline(BaseTransform):
    """Add a polynomial "baseline" to an existing curve

    Physically, a baseline acts as an indeterministic
    systematic drift from true zero which may be induced
    by a measurement device or otherwise.
    """

    def __init__(self,
                 intercept: BaseDistribution,
                 linear: BaseDistribution,
                 quadratic: BaseDistribution,
                 x0: BaseDistribution,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intercept_distribution = intercept
        self.linear_const_distribution = linear
        self.quadratic_const_distribution = quadratic
        self.translation_distribution = x0

    def transform(self, series, bboxes) -> Augmentation:
        a = self.intercept_distribution.gen_value(self._deterministic)
        b = self.linear_const_distribution.gen_value(self._deterministic)
        c = self.quadratic_const_distribution.gen_value(self._deterministic)
        x0 = self.translation_distribution.gen_value(self._deterministic)
        x_ = numpy.linspace(0, 1, series.shape[-1])
        baseline = a + b*(x_ - x0) + c*(x_ - x0)**2
        self._cache.update({
            "a": a,
            "b": b,
            "c": c,
            "x0": x0,
            "baseline": baseline,
        })
        new_series = series + baseline[numpy.newaxis,:]
        return Augmentation(series=new_series, bboxes=bboxes)


class Translate(BaseTransform):
    """Translate a curve left or right

    Extrapolates values after translation on edges
    """

    def __init__(self,
                 x_shift: BaseDistribution,
                 extrapolation_method: str,
                 *args, **kwargs):
        SUPPORTED_EXTRAPOLATIONS = {
            "constant": self._constant_extrapolation
        }
        super().__init__(*args, **kwargs)
        self.x_shift = x_shift
        self.extrapolation_method = SUPPORTED_EXTRAPOLATIONS.get(extrapolation_method, None)
        if self.extrapolation_method is None:
            raise NotImplementedError(
                f"Supplied extrapolation method ({extrapolation_method})"
                " is not implemented.")

    def _constant_extrapolation(self, series, nbins):
        return series[..., 0]

    def __extrapolate_left(self, series, nbins):
        series[..., :nbins] = self.extrapolation_method(series[..., nbins:], nbins)
        return series

    def __roll_with_extrapolation(self, series, shifts):
        if shifts == 0:
            return series
        elif shifts < 0:
            # for negative rolls we do the same on a reversed list
            series = series[..., ::-1]
        new_series = numpy.zeros(series.shape)
        new_series[..., abs(shifts):] = series[..., :-abs(shifts)]
        new_series = self.__extrapolate_left(new_series, nbins=abs(shifts))
        if shifts < 0:
            # the reversed list now has to be reversed back
            new_series = new_series[...,::-1]
        return new_series

    def transform(self, series: torch.Tensor, bboxes) -> Augmentation:
        series_len = series.shape[-1]
        x_shift = min(max(self.x_shift.gen_value(), -0.8), 0.8) # limit x_shift to +/- 0.8
        x_shift_bins = int(x_shift*series_len)
        x_shift_perc = x_shift_bins/series_len
        new_series = self.__roll_with_extrapolation(series, x_shift_bins)
        self._cache.update({
            "x_shift": x_shift,
            "x_shift_bins": x_shift_bins,
            "x_shift_perc": x_shift_perc,
        })

        new_bboxes = []
        for bbox in bboxes:
            mean, width, cls_ = bbox
            tmp_mean = mean + x_shift_perc
            x0 = max(tmp_mean - width/2, 0)
            x1 = min(tmp_mean + width/2, 1)
            # cut off oversized bbox mean and width
            new_mean = (x0 + x1)/2
            new_width = x1 - x0

            new_bboxes.append((new_mean, new_width, cls_))

        return Augmentation(series=new_series, bboxes=new_bboxes)


class Flip(BaseTransform):
    """Translate a curve left or right

    Extrapolates values after translation on edges
    """

    def transform(self, series, bboxes) -> Augmentation:
        new_series = series[..., ::-1]
        new_bboxes = [(1-mean, width, cls_)
                      for (mean, width, cls_) in bboxes]
        return Augmentation(new_series, new_bboxes)


class Stretch(BaseTransform):
    # TODO
    # stretch by a given factor
    # negative stretch is squeezing
    # probably implementable with cubic spline interpolation
    # and resampling based on x -> squeeze_factor * (x - x0)
    pass

# TODO scale transform: scale y


if __name__ == "__main__":

    def gaussian(x, mu, sig, amplitude):
        return amplitude * numpy.exp(-(x-mu)**2/(2*sig**2))

    import matplotlib.pyplot as plt

    numpy.random.seed(42)
    for i in range(10):
        x = numpy.linspace(0, 1, 416)
        amplitude = numpy.random.normal(0.5, 0.1)
        mu = numpy.random.normal(0.5, 0.1)
        sig = numpy.random.uniform(0.01, 0.05)
        amplitude = numpy.random.normal(0.5, 0.15)
        series = (gaussian(x, mu, sig, amplitude))[numpy.newaxis, :]

        xmax = min(mu + 3*sig, 1)
        xmin = max(mu - 3*sig, 0)
        bbox_mean = (xmax + xmin)/2
        bbox_width = xmax - xmin
        cls_ = 0
        bboxes = numpy.array([[bbox_mean, bbox_width, 0]])

        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        # plot original
        for ax in (ax0, ax1, ax2, ax3):
            ax.plot(x, series[0, :], label="original", c="C0")
            for bbox in bboxes:
                mean, width, cls_ = bbox
                ax.axvspan(mean-width/2, mean+width/2, alpha=0.5, color="C0")

        # plot translation
        ax0.set_title("Translate transform")
        translate_transform = Translate(x_shift=NormalDistribution(0, 0.2),
                                        extrapolation_method="constant",
                                        p=1)
        augmentation = translate_transform(series, bboxes)
        dx = translate_transform._cache["x_shift_perc"]
        ax0.plot(x, augmentation["series"][0, :], c="C1", label="transformed")
        ax0.axvline(mu, c="C0", ls="--")
        ax0.axvline(mu+dx, c="C1", ls="--")
        ax0.annotate("", xy=(mu+dx, 0.1), xytext=(mu, 0.1), arrowprops=dict(width=2,
                                                                            headwidth=6,
                                                                            headlength=6,
                                                                            shrink=0.1))
        new_bboxes = augmentation["bboxes"]
        for bbox in new_bboxes:
            mean, width, cls_ = bbox
            ax0.axvspan(mean-width/2, mean+width/2, alpha=0.5, color="C1")

        # plot baseline transform
        ax1.set_title("Polynomial baseline transform")
        baseline_transform = PolynomialBaseline(
            intercept=NormalDistribution(0, 0.1),
            linear=NormalDistribution(0, 0.05),
            quadratic=NormalDistribution(0, 0.05),
            x0=NormalDistribution(0.5, 0.1),
            p=1
            )
        augmentation = baseline_transform(series, bboxes)
        baseline = baseline_transform._cache["baseline"]
        ax1.plot(x, augmentation["series"][0, :], c="C1", label="transformed")
        ax1.plot(x, baseline, ls="--", lw=1, c="black", label="baseline")
        new_bboxes = augmentation["bboxes"]
        for bbox in new_bboxes:
            mean, width, cls_ = bbox
            ax1.axvspan(mean-width/2, mean+width/2, alpha=0.5, color="C1")

        # plot flip transform
        flip_transform = Flip(p=1)
        ax2.set_title("Flip transform")
        augmentation = flip_transform(series, bboxes)
        ax2.plot(x, augmentation["series"][0, :], c="C1", label="transformed")
        new_bboxes = augmentation["bboxes"]
        for bbox in new_bboxes:
            mean, width, cls_ = bbox
            ax2.axvspan(mean-width/2, mean+width/2, alpha=0.5, color="C1")
            ax2.axvline(0.5, ls=":", lw=1, color="black")

        for ax in (ax0, ax1, ax2):
            ax.legend()

        # plot composed transform
        composed_transform = ComposedTransform(transforms=[translate_transform,
                                                           baseline_transform,
                                                           flip_transform],
                                               p=1)
        ax3.set_title("Composed transform")
        augmentation = composed_transform(series, bboxes)
        new_bboxes = augmentation["bboxes"]
        # for i, (aug_key, aug_cache) in enumerate(composed_transform._cache.items(),
        #                                          start=1):
        i = 1
        for (aug_key, aug_cache) in composed_transform._cache.items():
            augmentation = aug_cache.get("augmentation")
            ax3.plot(x, augmentation["series"][0, :], c=f"C{i}", label=aug_key)
            new_bboxes = augmentation["bboxes"]
            for bbox in new_bboxes:
                mean, width, cls_ = bbox
                ax3.axvspan(mean-width/2, mean+width/2, alpha=0.5, color=f"C{i}")
            i+=1

        for ax in (ax0, ax1, ax2, ax3):
            ax.legend()
        plt.show()
        plt.show()