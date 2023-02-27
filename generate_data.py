from option import BaseOptionParser
import scipy.io as scio
import torch
import json
import numpy as np
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
import pickle


class SMPLModel():
    def __init__(self, model_path):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters
        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='iso-8859-1')
            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])
            self.v_template = np.asarray(params['v_template'])
            self.shapedirs = np.asarray(params['shapedirs'])
            self.faces = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.faces.dtype='int32'
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.theta_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.theta = np.zeros(self.theta_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None
        self.G = None
        self.joints = None

        self.update()

    def set_params(self, theta=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        theta: Parameter for model pose. A [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if theta is not None:
            self.theta = theta
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        theta_cube = self.theta.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(theta_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        g = np.empty((self.kintree_table.shape[1], 4, 4))
        g[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            g[i] = g[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        self.joints = g[:, :3, 3] + self.trans

        # remove the transformation due to the rest pose
        G = g - self.pack(
            np.matmul(
                g,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        self.G = G

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ---------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.
    
        """
        print("save")
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.f + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def compute_vertex_normal(vertices, indices):
    # code obtained from https://github.com/BachiLi/redner
    # redner/pyredner/shape.py
    def dot(v1, v2):
        # v1 := 13776 x 3
        # v1 := 13776 x 3
        # return := 13776

        return torch.sum(v1 * v2, dim=1)

    def squared_length(v):
        # v = 13776 x 3
        return torch.sum(v * v, dim=1)

    def length(v):
        # v = 13776 x 3
        # 13776
        return torch.sqrt(squared_length(v))

    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    # vertices := 6890 x 3
    # indices := 13776 x 3
    normals = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)
    v = [vertices[indices[:, 0].long(), :],
         vertices[indices[:, 1].long(), :],
         vertices[indices[:, 2].long(), :]]

    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / torch.reshape(e1_len, [-1, 1])  # 13776, 3
        side_b = e2 / torch.reshape(e2_len, [-1, 1])  # 13776, 3
        if i == 0:
            n = torch.cross(side_a, side_b)  # 13776, 3
            n = n / torch.reshape(length(n), [-1, 1])
        angle = torch.where(dot(side_a, side_b) < 0,
                            np.pi - 2.0 * torch.asin(0.5 * length(side_a + side_b)),
                            2.0 * torch.asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)  # 13776

        # XXX: Inefficient but it's PyTorch's limitation
        contrib = n * (sin_angle / (e1_len * e2_len)).reshape(-1, 1).expand(-1, 3)  # 13776, 3
        index = indices[:, i].long().reshape(-1, 1).expand([-1, 3])  # torch.Size([13776, 3])
        normals.scatter_add_(0, index, contrib)

    normals = normals / torch.reshape(length(normals), [-1, 1])
    return normals.contiguous()


def rotate_mesh(mesh_v, angle):

    angle = np.radians(angle)
    # rz = np.array([
    #     [np.cos(angle), -np.sin(angle), 0. ],
    #     [np.sin(angle), np.cos(angle), 0. ],
    #     [0., 0., 1. ]
    # ])
    rx = np.array([
        [1., 0., 0.],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle), ],
    ])
    return rx.dot(mesh_v.T).T


def convert2matrix(x):
    '''
    In order to preprocess data, convert list of tuple to matrix
    
    Parameter:
    ---------
    x: a list whose element is a tuple

    Return:
    ------
    y: a matrix corresponding to x
    '''
    x = list(x)
    y = []
    for i in range(len(x)):
        y.append(x[i])
    y = np.array(y)
    # print(y.shape)
    
    return y


def split_train_val_test(length, train_ratio, val_ratio, **data):
    ''' Split data with same length to train, val and test set '''
    parser = BaseOptionParser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    shuffled_indices = np.random.permutation(length)

    train_size = int(length * train_ratio)
    val_size = int(length * val_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]
    
    train_data = {}
    val_data = {}
    test_data = {}

    for key, value in data.items():
        train_data[key] = value[train_indices]
        val_data[key] = value[val_indices]
        test_data[key] = value[test_indices]

    return train_data, val_data, test_data


def generate_data():
    '''
    This function loads thetas, betas, genders, joint coordinates and generates markers and their labels from surreal dataset
    '''
    parser = BaseOptionParser()
    args = parser.parse_args()

    # get markerset information and randomly choose one
    np.random.seed(args.seed)
    m2b_distance = 0.0095
    with open('./ssm_all_marker_placements.json') as f:
        all_marker_placements = json.load(f)
    # all_mrks_keys = list(all_marker_placements.keys())
    # for key in all_mrks_keys:
    #     print(key, len(all_marker_placements[key]))
    # chosen_k = all_mrks_keys[np.random.choice(len(all_marker_placements))]
    chosen_k = '20160930_50032_ATUSquat_sync'
    # print(chosen_k)
    chosen_marker_set = all_marker_placements[chosen_k]
    label = list(chosen_marker_set.keys())
    m = len(chosen_marker_set)

    # define the path of info file in surreal dataset and the path to save markers and poses
    for name in ['train', 'val', 'test']:
        dataset = {}
        markers = []
        thetas = []
        betas = []
        genders = []
        joints = []

        data_path = glob.glob(os.path.join(args.basic_path, 'cmu', name, 'run1', '*', '*info.mat'))   
        data_path.sort() 
        # print(data_path)

        # load SMPL model and face information
        model_path = args.basic_path + 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'   # the path of SMPL model 
        model = SMPLModel(model_path)
        face = model.faces                                                      # (13776, 3)
        # print(face.shape)

        # generate marker and pose files in specific range
        for i in range(len(data_path)):
            subject = data_path[i].split('/')[-2]
            seq = data_path[i].split('/')[-1].rsplit('_')[-2]
            print('Processing the data for sequence {} of subject {} in {} set...'.format(seq, subject, name))

            database = scio.loadmat(data_path[i])
            beta = np.array(database['shape'].T)                            # shape parameters with size of (f, 10)
            theta = np.array(database['pose'].T)                            # pose parameters with size of (f, 72)
            gender = np.array(database['gender'])                           # 0: 'female', 1: 'male', gender with size of (f, 1)
            joint = np.array(database['joints3D'].T)                        # 3D coordinates of joints with size of (f, 24, 3)
            if joint.ndim != 3:
                joint = joint[:, :, None]
            # print(beta.shape, theta.shape, gender.shape, joint.shape)

            f = theta.shape[0]
            marker = np.zeros((f, m, 3))                                    # (f, m, 3)

            # add noise
            cur_m2b_distance = m2b_distance + abs(np.random.normal(0, m2b_distance / 3., size=[3])) 

            for fIdx in range(f):
                model.theta[:] = theta[fIdx, :].reshape(24, 3)
                model.beta[:] = beta[fIdx, :]
                model.update()
                vertex = rotate_mesh(model.verts, 90)
                vertex = model.verts

                vn = compute_vertex_normal(torch.Tensor(vertex), torch.Tensor(face))

                for mrk_id, vid in enumerate(chosen_marker_set.values()):
                    marker[fIdx, mrk_id, :] = torch.Tensor(vertex[vid]) + torch.Tensor(cur_m2b_distance) * vn[vid]
                
            print('Successfully generate marker for the No.{} file of total {} files in {} set!'.format(
                i+1, len(data_path), name))    

            marker = marker.astype(np.float32)  
            marker = marker.reshape(f, -1)                                      # (f, m*3)
            joint = joint.reshape(f, -1)                                        # (f, 24*3)

            # print(marker.shape, theta.shape, beta.shape, gender.shape, joint.shape)
            markers.append(marker)
            thetas.append(theta)
            betas.append(beta)
            genders.append(gender)
            joints.append(joint)

        marker = np.vstack(markers)                                         # (f, m*3)
        marker = marker.reshape(-1, m, 3)                                   # (f, m, 3)
        theta = np.vstack(thetas)                                           # (f, 72)
        theta = theta.reshape(-1, 24, 3)                                    # (f, 24, 3)
        beta = np.vstack(betas)                                             # (f, 10)
        gender = np.vstack(genders)                                         # (f, 1)
        joint = np.vstack(joints)                                           # (f, 24*3)
        joint = joint.reshape(-1, 24, 3)                                    # (f, 24, 3)

        # print(marker.shape, theta.shape, beta.shape, gender.shape, joint.shape)

        dataset['label'] = label
        dataset['marker'] = marker
        dataset['theta'] = theta
        dataset['beta'] = beta
        dataset['gender'] = gender
        dataset['joint'] = joint

        np.save(os.path.join(args.data_path, name + '_' + str(m) + '.npy'), dataset)
        print('Successfully save {} data, and the total number of frames is {}!'.format(name, marker.shape[0]))


if __name__ == '__main__':
    generate_data()
    # save2file()