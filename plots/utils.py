import numpy as np


from scipy.spatial import Delaunay

def fit(X, y):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    return gpr

def is_in(pt, points):
    hull = Delaunay(points)
    return hull.find_simplex(pt) >= 0

def load_stat(args):
    stat_path = "/".join(args.model_path.split("/")[:-1])+"/stat0.npz"
    s = np.load(stat_path,allow_pickle=True)
    return s["in_means"], s["in_stds"], s["out_means"], s["out_stds"]




def in_hull(queries, equations):
    return np.all(queries @ equations[:-1] < - equations[-1], axis=1)

# # ============== Demonstration ================
#
# points = np.random.rand(8, 2)
# queries = np.random.rand(3, 2)
# print(in_hull(points, queries))
def sample_from(set1, num_points, sel_indices=None, hull_sampling=False, gain=1, faster_hull=False):
    if sel_indices:
        s_mins = np.min(set1, axis=0)[sel_indices]
        s_maxs = np.max(set1, axis=0)[sel_indices]
        pts_list = np.zeros((0, len(sel_indices)))
        #if faster_hull:
            # hull1 = _Qhull(b"i", set1[:, sel_indices],
            #               options=b"",
            #               furthest_site=False,
            #               incremental=False,
            #               interior_point=None)
            #equations = hull1.get_simplex_facet_array()[2].T
        #else:
        hull1 = Delaunay(set1[:, sel_indices])
    else:
        s_mins = np.min(set1, axis=0)
        s_maxs = np.max(set1, axis=0)
        pts_list = np.zeros((0, set1.shape[1]))
        hull1 = Delaunay(set1)
    while pts_list.shape[0]<num_points:
        samples = np.random.random((int(num_points * gain), s_mins.shape[0]))
        points = samples * (s_maxs - s_mins) + s_mins
        #if faster_hull:
            #ind = np.where(in_hull(points, equations) >= 0)[0]
        #else:
        ind = np.where(hull1.find_simplex(points) >= 0)[0]
        real_pts = points[ind]
        pts_list = np.concatenate((pts_list, real_pts), axis=0)
        pts_list = pts_list[:num_points]
    return pts_list, hull1


def cmp_prob(estimator, x1, x_min1, x_max1, y_min1, y_max1, sampled_x2, num_points, t):
    samples_all = sampled_x2

    s_mins2 = np.min(sampled_x2, axis=0)
    s_maxs2 = np.max(sampled_x2, axis=0)

    s_mins1=np.array(s_mins2)
    s_mins1[0] = x_min1
    s_mins1[1] = y_min1

    s_maxs1=np.array(s_maxs2)
    s_maxs1[0] = x_max1
    s_maxs1[1] = y_max1
    dim = s_mins1.shape[0]
    ind_list=[]
    for i in range(dim):
        ind_list.append(np.logical_and(samples_all[:, i]<=s_maxs1[i], samples_all[:,i]>s_mins1[i]))
    ind = np.stack(ind_list, axis=-1)
    ind = np.where(np.sum(ind,axis=-1)==dim)
    indices1 = ind[0]

    print(indices1.shape[0], samples_all.shape[0])
    if indices1.shape[0]==0:
        return 0.0

    samples1 = samples_all[indices1]
    samples2 = samples_all

    dens1 = estimator.predict(samples1)
    dens2 = estimator.predict(samples2)

    sum1 = np.sum(np.exp(dens1 * t))
    sum2 = np.sum(np.exp(dens2 * t))

    prob = sum1 / sum2

    return prob


def compute_prob(estimator, set1, set2, num_points, dim, sampled_points=None, hull2=None):
    if set1.shape[0]<=dim:
        return 0.0
    # set1 an array of points
    # set2 an array of points
    hull1 = Delaunay(set1)
    if hull2 is None:
        hull2 = Delaunay(set2)
    if sampled_points is None:
        s2_min = np.min(set2, axis=0)
        s2_max = np.max(set2, axis=0)
        xs2 = np.random.random((num_points, set2.shape[1]))
        xs2 = xs2 * (s2_max - s2_min) + s2_min
    else:
        xs2 = sampled_points
    real_xs2 = xs2
    ind1 = np.where(hull1.find_simplex(real_xs2)>=0)[0]
    real_xs1 = real_xs2[ind1]
    if real_xs1.shape[0] == 0:
        return 0.0
    dens1 = estimator(*(real_xs1.T))
    dens2 = estimator(*(real_xs2.T))
    sum1 = np.sum(dens1)
    sum2 = np.sum(dens2)
    prob = sum1 / sum2
    return prob