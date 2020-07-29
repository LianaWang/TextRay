import numpy as np
import numpy.polynomial.chebyshev as chebyshev
import numpy.polynomial.polynomial as polynomial
from shapely.geometry import Polygon, LineString, MultiLineString, GeometryCollection
import math


def polar_coord(point, center):
    """
    from cartesian coordinates to polar coordinates
    :param point: [k, 2] {x, y}
    :param center: {x, y}
    :return: polar coordinates
    """
    x = point[0] - center[0]
    y = point[1] - center[1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.array([phi, rho])


def cart_coord(contours, centers):
    """ from polar coordinates to cartesian coordinates
    :param contours: [K, 2] {theta, r}
    :param centers:
    :return: cartesian coordinates
    """
    cart_contours = []
    for i in range(len(contours)):
        contour = contours[i]
        center = centers[i]
        cart_contour = np.zeros((contour.shape[0], contour.shape[1]))
        for j in range(contour.shape[0]):
            cart_contour[j, 0] = contour[j, 1] * math.cos(contour[j, 0]) + center[0]
            cart_contour[j, 1] = contour[j, 1] * math.sin(contour[j, 0]) + center[1]
        cart_contours.append(cart_contour)
    return cart_contours


def sample_contour(points, centers, hard_flg, num_samples=360):
    R_ = 9999.0
    contours = []
    skeleton = []
    # trans = np.floor(np.array(trans) / 3.0)
    for ix in range(len(points)):
        if hard_flg[ix] == 1:
            continue
        poly = Polygon(points[ix].reshape([-1, 2]))
        if not poly.is_valid:
            print("polygon not valid")
            hard_flg[ix] = 1
            continue
        center = np.array(centers[ix])
        edge_rt = []
        edge_xy = []
        intersects = []
        theta_d = np.linspace(-1, 1, num_samples, endpoint=False)
        for theta in theta_d:
            inter_x = R_ * math.cos(theta * math.pi) + center[0]
            inter_y = R_ * math.sin(theta * math.pi) + center[1]
            intersects.append([inter_x, inter_y])
        valid = True
        for p in range(len(intersects)):
            intersect = intersects[p]
            line = LineString([(center[0], center[1]), (intersect[0], intersect[1])])
            edge_cart = None
            if type(poly.intersection(line)) is LineString:
                edge_cart = np.rint(list(poly.intersection(line).coords)[-1])
            elif type(poly.intersection(line)) is MultiLineString:
                edge_cart = np.rint(list(poly.intersection(line).geoms[-1].coords)[-1])
            elif type(poly.intersection(line)) is GeometryCollection:
                edge_cart = np.rint(list(poly.intersection(line).geoms[-1].coords)[-1])
            if edge_cart is None:
                print("intersection not valid", type(poly.intersection(line)))
                valid = False
                hard_flg[ix] = 1
                continue
            edge_pol = polar_coord(edge_cart, center)
            edge_rt.append(edge_pol)
            edge_xy.append(edge_cart)
        if valid:
            contours.append(np.array(edge_rt))
            skeleton.append(np.array(edge_xy))
        else:
            print("edge_cart is None", poly)
    if len(contours) == 0:
        print("contours is zero")
    return contours, skeleton, hard_flg


def find_principle(contours_xy):
    p = []
    for contour in contours_xy:
        polygon = Polygon(contour)
        # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
        mbr_points = list(zip(*polygon.minimum_rotated_rectangle.exterior.coords.xy))
        x = np.array(polygon.minimum_rotated_rectangle.exterior.coords.xy[0])
        y = np.array(polygon.minimum_rotated_rectangle.exterior.coords.xy[1])
        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
        minor = np.argmin(mbr_lengths)
        major = np.argmax(mbr_lengths)
        minor_axis,_ = polar_coord([x[minor+1] - x[minor], y[minor+1] - y[minor]], [0,0])
        major_axis,_ = polar_coord([x[major+1] - x[major], y[major+1] - y[major]], [0,0])
        if minor_axis < 0:
            minor_axis += math.pi
        elif minor_axis == math.pi:
            minor_axis = 0
        assert(0 <= minor_axis < math.pi)
        p.append(minor_axis/math.pi)
    return p


def rotate_cheby_fit(contours, skeleton_ori, num_coefs):
    p = find_principle(skeleton_ori)
    coefs, r_maxs = [], []
    for ix, contour in enumerate(contours):
        contour = np.array(contour)
        start = min(max(int(p[ix]*180.0), 0), 179)
        contour = np.vstack((contour[start:, :], contour[:start, :]))
        theta = np.linspace(-1, 1, 360, endpoint=False)
        r = contour[:, 1]
        r_max = np.max(r)
        r = r / r_max
        coef, res = chebyshev.chebfit(theta, r, num_coefs, full=True)
        coefs.append(coef)
        r_maxs.append(r_max)
    coefs = np.array(coefs)
    p = np.array(p)
    r_maxs = np.array(r_maxs)
    cheby_coef = np.hstack((coefs, p[:, np.newaxis], r_maxs[:, np.newaxis]))
    return cheby_coef
    

def cheby_fit(contours, num_coefs):
    coefs, r_maxs = [], []
    for contour in contours:
        contour = np.array(contour)
        theta = np.linspace(-1, 1, 360, endpoint=False)
        r = contour[:, 1]
        r_max = np.max(r)
        r = r / r_max
        coef, res = chebyshev.chebfit(theta, r, num_coefs, full=True)
        coefs.append(coef)
        r_maxs.append(r_max)
    coefs = np.array(coefs)
    r_maxs = np.array(r_maxs)
    cheby_coef = np.hstack((coefs, r_maxs[:, np.newaxis]))
    return cheby_coef


def rotate_fourier_fit(contours, skeleton_ori, num_coefs):
    p = find_principle(skeleton_ori)
    coefs, r_maxs = [], []
    for ix, contour in enumerate(contours):
        contour = np.array(contour)
        start = min(max(int(p[ix]*180.0), 0), 179)
        contour = np.vstack((contour[start:, :], contour[:start, :]))
        r = contour[:, 1]
        r_max = np.max(r)
        r = r / r_max
        coef = np.fft.fft(r, num_coefs)
        coef = np.concatenate([coef.real, coef.imag])
        coefs.append(coef)
        r_maxs.append(r_max)
    coefs = np.array(coefs)
    p = np.array(p)
    r_maxs = np.array(r_maxs)
    fori_coef = np.hstack((coefs, p[:, np.newaxis], r_maxs[:, np.newaxis]))
    return fori_coef


def fourier_fit(contours, num_coefs):
    """
    :param contours:
    :param num_coefs:
    :return: [K, 2 * num_coefs + 1]
    """
    coefs, r_maxs = [], []
    for contour in contours:
        contour = np.array(contour)
        r = contour[:, 1]
        r_max = np.max(r)
        r = r / r_max
        coef = np.fft.fft(r, num_coefs)
        coef = np.concatenate([coef.real, coef.imag])
        coefs.append(coef)
        r_maxs.append(r_max)

    coefs = np.array(coefs)
    r_maxs = np.array(r_maxs)
    cheby_coef = np.hstack((coefs, r_maxs[:, np.newaxis]))
    return cheby_coef


def poly_fit(contours, num_coefs):
    coefs, r_maxs = [], []
    for contour in contours:
        contour = np.array(contour)
        theta = np.linspace(-1, 1, 360, endpoint=False)
        r = contour[:, 1]
        r_max = np.max(r)
        r = r / r_max
        coef, res = polynomial.polyfit(theta, r, num_coefs, full=True)
        coefs.append(coef)
        r_maxs.append(r_max)
    coefs = np.array(coefs)
    r_maxs = np.array(r_maxs)
    poly_coef = np.hstack((coefs, r_maxs[:, np.newaxis]))
    return poly_coef


def expand_twelve(vertices):
    """
    expand vetices of even dimensions to 12 points, used in baseline model
    :param vertices: [npts, 2]
    :return:
    """
    box = np.zeros((12, 2), dtype=np.float32)
    p = vertices.shape[0]
    if p == 4:
        box[0, :] = vertices[0, :]
        box[1, :] = [(4 * vertices[0, 0] + vertices[1, 0]) / 5.0, (4 * vertices[0, 1] + vertices[1, 1]) / 5.0]
        box[2, :] = [(3 * vertices[0, 0] + 2 * vertices[1, 0]) / 5.0,
                     (3 * vertices[0, 1] + 2 * vertices[1, 1]) / 5.0]
        box[3, :] = [(2 * vertices[0, 0] + 3 * vertices[1, 0]) / 5.0,
                     (2 * vertices[0, 1] + 3 * vertices[1, 1]) / 5.0]
        box[4, :] = [(vertices[0, 0] + 4 * vertices[1, 0]) / 5.0, (vertices[0, 1] + 4 * vertices[1, 1]) / 5.0]
        box[5, :] = vertices[1, :]
        box[6, :] = vertices[2, :]
        box[7, :] = [(vertices[3, 0] + 4 * vertices[2, 0]) / 5.0, (vertices[3, 1] + 4 * vertices[2, 1]) / 5.0]
        box[8, :] = [(2 * vertices[3, 0] + 3 * vertices[2, 0]) / 5.0,
                     (2 * vertices[3, 1] + 3 * vertices[2, 1]) / 5.0]
        box[9, :] = [(3 * vertices[3, 0] + 2 * vertices[2, 0]) / 5.0,
                     (3 * vertices[3, 1] + 2 * vertices[2, 1]) / 5.0]
        box[10, :] = [(4 * vertices[3, 0] + vertices[2, 0]) / 5.0, (4 * vertices[3, 1] + vertices[2, 1]) / 5.0]
        box[11, :] = vertices[3, :]
    elif p == 6:
        box[0, :] = vertices[0, :]
        box[1, :] = [(vertices[0, 0] + vertices[1, 0]) / 2.0, (vertices[0, 1] + vertices[1, 1]) / 2.0]
        box[2, :] = vertices[1, :]
        box[3, :] = [(2 * vertices[1, 0] + vertices[2, 0]) / 3.0, (2 * vertices[1, 1] + vertices[2, 1]) / 3.0]
        box[4, :] = [(vertices[1, 0] + 2 * vertices[2, 0]) / 3.0, (vertices[1, 1] + 2 * vertices[2, 1]) / 3.0]
        box[5, :] = vertices[2, :]
        box[6, :] = vertices[3, :]
        box[7, :] = [(vertices[4, 0] + 2 * vertices[3, 0]) / 3.0, (vertices[4, 1] + 2 * vertices[3, 1]) / 3.0]
        box[8, :] = [(2 * vertices[4, 0] + vertices[3, 0]) / 3.0, (2 * vertices[4, 1] + vertices[3, 1]) / 3.0]
        box[9, :] = vertices[4, :]
        box[10, :] = [(vertices[5, 0] + vertices[4, 0]) / 2.0, (vertices[5, 1] + vertices[4, 1]) / 2.0]
        box[11, :] = vertices[5, :]
    elif p == 8:
        box[0, :] = vertices[0, :]
        box[1, :] = [(vertices[0, 0] + vertices[1, 0]) / 2.0, (vertices[0, 1] + vertices[1, 1]) / 2.0]
        box[2, :] = vertices[1, :]
        box[3, :] = vertices[2, :]
        box[4, :] = [(vertices[2, 0] + vertices[3, 0]) / 2.0, (vertices[2, 1] + vertices[3, 1]) / 2.0]
        box[5, :] = vertices[3, :]
        box[6, :] = vertices[4, :]
        box[7, :] = [(vertices[4, 0] + vertices[5, 0]) / 2.0, (vertices[4, 1] + vertices[5, 1]) / 2.0]
        box[8, :] = vertices[5, :]
        box[9, :] = vertices[6, :]
        box[10, :] = [(vertices[6, 0] + vertices[7, 0]) / 2.0, (vertices[6, 1] + vertices[7, 1]) / 2.0]
        box[11, :] = vertices[7, :]
    elif p == 10:
        box[0, :] = vertices[0, :]
        box[1, :] = vertices[1, :]
        box[2, :] = vertices[2, :]
        box[3, :] = vertices[3, :]
        box[4, :] = [(vertices[3, 0] + vertices[4, 0]) / 2.0, (vertices[3, 1] + vertices[4, 1]) / 2.0]
        box[5, :] = vertices[4, :]
        box[6, :] = vertices[5, :]
        box[7, :] = [(vertices[5, 0] + vertices[6, 0]) / 2.0, (vertices[5, 1] + vertices[6, 1]) / 2.0]
        box[8, :] = vertices[6, :]
        box[9, :] = vertices[7, :]
        box[10, :] = vertices[8, :]
        box[11, :] = vertices[9, :]
    elif p == 12:
        box = vertices
    else:
        raise ValueError(f'Invalid points dimension: {p}')

    return box


def inner_center(vertices):
    p = vertices.shape[0]
    if p == 4:
        x = np.sum(vertices[:, 0]) / 4.0
        y = np.sum(vertices[:, 1]) / 4.0
    elif p == 6:
        x = np.sum(vertices[[1, 4], 0]) / 2.0
        y = np.sum(vertices[[1, 4], 1]) / 2.0
    elif p == 8:
        x = np.sum(vertices[[1, 2, 5, 6], 0]) / 4.0
        y = np.sum(vertices[[1, 2, 5, 6], 1]) / 4.0
    elif p == 10:
        x = np.sum(vertices[[2, 7], 0]) / 2.0
        y = np.sum(vertices[[2, 7], 1]) / 2.0
    elif p == 12:
        x = np.sum(vertices[[2, 3, 8, 9], 0]) / 4.0
        y = np.sum(vertices[[2, 3, 8, 9], 1]) / 4.0
    else:
        raise AssertionError('vertices must have 4/6/8/10/12 points')
    return np.array([x, y])
