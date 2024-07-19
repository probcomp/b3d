import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(jnp.vectorize, signature="(k)->(k)")
def rgb_to_lab(rgb):
    # Convert sRGB to linear RGB
    rgb = jnp.clip(rgb, 0, 1)
    mask = rgb > 0.04045
    rgb = jnp.where(mask, jnp.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # RGB to XYZ
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    rgb_to_xyz = jnp.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = jnp.dot(rgb, rgb_to_xyz.T)

    # XYZ to LAB
    # https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])  # D65 white point
    xyz_normalized = xyz / xyz_ref
    mask = xyz_normalized > 0.008856
    xyz_f = jnp.where(
        mask, jnp.power(xyz_normalized, 1 / 3), 7.787 * xyz_normalized + 16 / 116
    )

    L = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])

    lab = jnp.stack([L, a, b], axis=-1)
    return lab



@partial(jnp.vectorize, signature="(k)->(k)")
def lab_to_rgb(lab):
    D65 = (0.95047, 1.00000, 1.08883)  # Reference white
    y = (lab[..., 0] + 16.0) / 116.0
    x = lab[..., 1] / 500.0 + y
    z = y - lab[..., 2] / 200.0
    
    xyz = jnp.stack([x, y, z], axis=-1)
    mask = xyz > 0.2068966
    xyz = jnp.where(mask, xyz ** 3, (xyz - 16.0 / 116.0) / 7.787)
    
    xyz = xyz * jnp.array(D65)
    mat = jnp.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb = jnp.tensordot(xyz, mat, axes=([-1], [1]))
    mask = rgb > 0.0031308
    rgb = jnp.where(mask, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055, 12.92 * rgb)
    return jnp.clip(rgb, 0, 1)



@partial(jnp.vectorize, signature="(k)->(k)")
def rgb_to_hsv(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    max_c = jnp.max(rgb)
    min_c = jnp.min(rgb)
    delta = max_c - min_c
    
    # Hue calculation
    def hue_func(r, g, b, max_c, delta):
        h = jnp.where(max_c == r, (g - b) / delta % 6, 0)
        h = jnp.where(max_c == g, (b - r) / delta + 2, h)
        h = jnp.where(max_c == b, (r - g) / delta + 4, h)
        return h * 60

    h = hue_func(r, g, b, max_c, delta)
    h = jnp.where(delta == 0, 0, h)
    
    # Saturation calculation
    s = jnp.where(max_c == 0, 0, delta / max_c)
    
    # Value calculation
    v = max_c
    
    return jnp.array([h, s, v])


def cie94_err(lab1, lab2, kC=1, kH=1, kL=1.0, K1=0.045, K2=0.015):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = jnp.sqrt(a1**2 + b1**2)
    C2 = jnp.sqrt(a2**2 + b2**2)

    deltaL = L1 - L2
    deltaa = a1 - a2
    deltab = b1 - b2
    deltaC_ab = C1 - C2
    deltaH_ab = jnp.sqrt(deltaa**2 + deltab**2 - deltaC_ab**2)  # can also use cie76

    SL = 1
    SC = 1 + K1 * C1
    SH = 1 + K2 * C1

    deltaE_94 = jnp.sqrt(
        (deltaL / (kL * SL)) ** 2
        + (deltaC_ab / (kC * SC)) ** 2
        + (deltaH_ab / (kH * SH)) ** 2
    )
    return deltaE_94


@partial(jnp.vectorize, signature="(n,a),(n,a)->(n)")
def ciede2000_err(I1, I2, kC=1, kH=1, kL=1.0, K1=0.045, K2=0.015):
    """
    Calculates the CIEDE2000 standard difference between
    n pairs of L*a*b* space colors

    In:
    _lab1: (n, 3) array of colors
    _lab2: (n, 3) array of colors

    Out:
    (n, 1) array of CIEDE2000 color difference between the input colors

    References:
    https://www.wikiwand.com/en/Color_difference#CIEDE2000
    """
    TWOFIVE_7 = np.asarray(6103515625, np.int64)  # 25**7

    C1 = jnp.sqrt(I1[..., 1] ** 2 + I1[..., 2] ** 2)
    C2 = jnp.sqrt(I2[..., 1] ** 2 + I2[..., 2] ** 2)
    Cbar = (C1 + C2) / 2

    G = 0.5 * (1 - (jnp.sqrt((Cbar**7) / (Cbar**7 + TWOFIVE_7))))

    a1 = (1 + G) * I1[..., 1]
    a2 = (1 + G) * I2[..., 1]
    C1d = jnp.sqrt(a1**2 + I1[..., 2] ** 2)
    C2d = jnp.sqrt(a2**2 + I2[..., 2] ** 2)

    # Calculating the Modified hue using the four-quadrant arctangent
    h1 = jnp.atan2(I1[..., 2], a1)
    h2 = jnp.atan2(I2[..., 2], a2)

    # Typically, these functions return an angular value in radians ranging
    # from -pi to pi. This must be converted to a hue angle in degrees between
    # 0° and 360° by addition of 2*pi to negative hue angles.
    h1 = jnp.where(h1 < 0, h1 + 2 * jnp.pi, h1)  # h1.at[h1<0].set(h1[h1<0] + 2*jnp.pi)
    h2 = jnp.where(h2 < 0, h2 + 2 * jnp.pi, h2)  # h2.at[h2<0].set(h2[h2<0] + 2*jnp.pi)
    h1 = jnp.where(
        (a1 == 0) & (I1[..., 2] == 0), 0, h1
    )  # h1.at[((a1 == 0) * (I1[...,2] == 0))].set(0)
    h2 = jnp.where(
        (a2 == 0) & (I2[..., 2] == 0), 0, h2
    )  # h2.at[((a2 == 0) * (I2[...,2] == 0))].set(0)

    ## Calculate dL, dC and dH
    # Here the following equations are laid to calculate the differences in
    # L, C and H values of the colors under consideration
    dL = I2[..., 0] - I1[..., 0]
    dC = C2d - C1d
    hsub = h2 - h1
    c1d_c2d_nonzero = C1d * C2d != 0
    hsub_under_pi = abs(hsub) <= jnp.pi
    dh = jnp.zeros(h2.shape)
    dh = jnp.where(c1d_c2d_nonzero & hsub_under_pi, hsub, dh)
    dh = jnp.where(c1d_c2d_nonzero & (hsub > jnp.pi), hsub - 2 * jnp.pi, dh)
    dh = jnp.where(c1d_c2d_nonzero & (hsub < -jnp.pi), hsub + 2 * jnp.pi, dh)
    dh = jnp.where(jnp.logical_not(c1d_c2d_nonzero), 0, dh)
    dH = 2 * jnp.sqrt(C1d * C2d) * jnp.sin(dh / 2)

    ## Calculate CIEDE2000 Color-Difference dE00:
    Lbar = (I1[..., 0] + I2[..., 0]) / 2
    Cdbar = (C1d + C2d) / 2
    hadd = h1 + h2
    hbar = jnp.zeros(h1.shape)
    hsub_under_pi = abs(hsub) <= jnp.pi
    hadd_under_2pi = hadd < 2 * jnp.pi
    hbar = jnp.where(hsub_under_pi & c1d_c2d_nonzero, hadd / 2, hbar)
    hbar = jnp.where(
        jnp.logical_not(hsub_under_pi) & hadd_under_2pi & c1d_c2d_nonzero,
        (hadd + 2 * jnp.pi) / 2,
        hbar,
    )
    hbar = jnp.where(
        jnp.logical_not(hsub_under_pi) & hadd_under_2pi & c1d_c2d_nonzero,
        (hadd - 2 * jnp.pi) / 2,
        hbar,
    )
    hbar = jnp.where(jnp.logical_not(c1d_c2d_nonzero), hadd, hbar)
    T = (
        1
        - 0.17 * jnp.cos(hbar - jnp.deg2rad(30))
        + 0.24 * jnp.cos(2 * hbar)
        + 0.32 * jnp.cos(3 * hbar + jnp.deg2rad(6))
        - 0.20 * jnp.cos(4 * hbar - jnp.deg2rad(63))
    )  # check the format for cos
    dTheta = jnp.deg2rad(30) * jnp.exp(
        -(((hbar - jnp.deg2rad(275)) / jnp.deg2rad(25)) ** 2)
    )
    RC = 2 * jnp.sqrt((Cdbar**7) / (Cdbar**7 + TWOFIVE_7))

    # Major parameters required to compute CIEDE2000:
    SL = 1 + ((K2 * (Lbar - 50) ** 2) / (jnp.sqrt(20 + (Lbar - 50) ** 2)))
    SC = 1 + (K1 * Cdbar)
    SH = 1 + (K2 * Cdbar * T)
    RT = -jnp.sin(2 * dTheta) * RC

    # CIEDE2000 Final Formula:
    dE = jnp.sqrt(
        (dL / (kL * SL)) ** 2
        + (dC / (kC * SC)) ** 2
        + (dH / (kH * SH)) ** 2
        + (RT * ((dC / (kC * SC)) * (dH / (kH * SH))))
    )

    return dE


def color_error_helper(observed_rgb, rendered_rgb, lab_tolerance, cielab=False):
    valid_data_mask = rendered_rgb.sum(-1) != 0.0
    # valid_data_mask = jnp.full(valid_data_mask.shape, True)
    observed_lab = rgb_to_lab(observed_rgb)
    rendered_lab = rgb_to_lab(rendered_rgb)
    if cielab:
        h, w, _ = observed_lab.shape
        error = ciede2000_err(
            observed_lab.reshape((h * w, 3)), rendered_lab.reshape((h * w, 3))
        ).reshape((h, w))
    else:
        error = jnp.linalg.norm(
            observed_lab[..., 1:3] - rendered_lab[..., 1:3], axis=-1
        ) + jnp.abs(observed_lab[..., 0] - rendered_lab[..., 0])
    inlier_match_mask = error < lab_tolerance
    inlier_match_mask = inlier_match_mask * valid_data_mask

    num_data_points = jnp.size(inlier_match_mask)
    num_inliers = jnp.sum(inlier_match_mask)
    num_no_data = jnp.sum(1.0 - valid_data_mask)
    num_outliers = num_data_points - num_inliers - num_no_data
    return (
        inlier_match_mask,
        num_data_points,
        num_inliers,
        num_no_data,
        num_outliers,
        error,
    )


if __name__ == "__main__":
    pureRed = jnp.array([[255, 0, 0], [255, 0, 0]])
    darkRed = jnp.array([[255, 10, 50], [255, 10, 50]])
    I1 = rgb_to_lab(pureRed / 255)
    I2 = rgb_to_lab(darkRed / 255)
    err = jax.jit(ciede2000_err)(I1, I2)
    print(err)
