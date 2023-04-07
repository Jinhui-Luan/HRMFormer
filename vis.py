from option import BaseOptionParser
from generate_data import SMPLModel, rotate_mesh, compute_vertex_normal
from transformer import Transformer, SMPLModel_torch
from main import get_data_loader, write_mesh, write_ply
import numpy as np
import torch
import json
import os
from tqdm import tqdm
import IPython


def gen_data():
    parser = BaseOptionParser()
    args = parser.parse_args()

    dataset = {}

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

    data = np.load('./anim1.npz')
    theta = data['poses'][:, :24, :]
    # beta = np.array([-2.2443552, 0.6893196, 2.108175, -0.57570285, -0.55536103, -1.5905423, -1.0977619, \
    #     -0.27627876, 0.59387493, -0.81694436])

    f = theta.shape[0]

    model_path = args.basic_path + 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'   # the path of SMPL model 
    model = SMPLModel(model_path)
    face = model.faces  
    
    marker = np.zeros((f, m, 3))  
    cur_m2b_distance = m2b_distance + abs(np.random.normal(0, m2b_distance / 3., size=[3])) 

    for fIdx in range(f):
        model.theta[:] = theta[fIdx, :, :].reshape(24, 3)
        # model.beta[:] = beta
        model.update()
        vertex = rotate_mesh(model.verts, 90)
        vertex = model.verts

        vn = compute_vertex_normal(torch.Tensor(vertex), torch.Tensor(face))

        for mrk_id, vid in enumerate(chosen_marker_set.values()):
            marker[fIdx, mrk_id, :] = torch.Tensor(vertex[vid]) + torch.Tensor(cur_m2b_distance) * vn[vid]
    
    marker = marker.astype(np.float32)
    beta = np.array([0, -3, 0, 0, 0, 0, 0, 0, 0, 0])
    beta = np.tile(beta, f).reshape(f, -1)
    # IPython.embed()
    
    dataset['marker'] = marker
    dataset['theta'] = theta
    dataset['beta'] = beta

    np.save(os.path.join(args.basic_path, 'dataset-amass', 'vis' + '_' + str(m) + '.npy'), dataset)


def test():
    # get parser of option.py
    parser = BaseOptionParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.d_h3 = args.d_model // 2
    args.d_h2 = args.d_h3 // 2
    args.d_h1 = args.d_h2 // 4
    args.d_ffn = args.d_model * 2

    model = Transformer(args).to(device)

    model_path = os.path.join(args.output_path, args.exp_name, 'models', 'model_best.chkpt')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print('Successfully load checkpoint of model!')
    dl_test = get_data_loader(args.basic_path, args.batch_size, 'vis', args.m, 1)

    # criterion = nn.MSELoss().to(device)
    smpl_model_path = os.path.join(args.basic_path, 'model_m.pkl')   
    smpl_model = SMPLModel_torch(smpl_model_path, device) 
    face = smpl_model.faces
    # print(face, face.shape)

    model.eval()
    # loss = []
    # loss_data = []
    # loss_joint = []
    # loss_vertex = []
    MPJPE = []
    MPVPE = []
    batch = 0

    desc = ' -       (Test) '
    with torch.no_grad():
        for data in tqdm(dl_test, mininterval=2, desc=desc, leave=False, ncols=100):
            marker = data['marker'].to(device)
            theta = data['theta'].to(device)
            beta = data['beta'].to(device)

            theta_pred = model(marker)

            smpl_model(beta, theta_pred)
            joint_pred = smpl_model.joints
            vertex_pred = smpl_model.verts

            smpl_model(beta, theta)
            joint = smpl_model.joints
            vertex = smpl_model.verts

            mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
            mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()

            # l_data = criterion(theta_pred, theta)
            # l_data = cal_data_loss(theta_pred, theta, args.rate, criterion)
            # l_joint = criterion(joint_pred, joint)
            # l_vertex = criterion(vertex_pred, vertex)
            # l = args.lambda1 * l_data + args.lambda2 * l_joint +  args.lambda3 * l_vertex

            # loss.append(l)
            # loss_data.append(l_data)
            # loss_joint.append(l_joint)
            # loss_vertex.append(l_vertex)
            MPJPE.append(mpjpe.clone().detach())
            MPVPE.append(mpvpe.clone().detach())

            if args.visualize:

                for i in range(marker.shape[0]):
                    # generate rgb color
                    rgb_marker = np.repeat(np.array([[255, 0, 0]]), marker.shape[1], axis=0)        # show marker in red 
                    rgb_joint = np.repeat(np.array([[0, 255, 0]]), joint.shape[1], axis=0)          # show joint in green
                    rgb_vertex = np.repeat(np.array([[123, 123, 123]]), vertex.shape[1], axis=0)    # show vertex in gray             

                    # concatenate the points of mesh and markers
                    m = marker[i].to('cpu')
                    j = joint[i].to('cpu')
                    j_pred = joint_pred[i].to('cpu')
                    v = vertex[i].to('cpu')
                    v_pred = vertex_pred[i].to('cpu')
                    
                    # point = np.vstack((j, v))
                    # point_pred = np.vstack((j_pred, v_pred))
                    # rgb = np.vstack((rgb_joint, rgb_vertex))

                    os.makedirs(os.path.join(args.vis_path, 'test'), exist_ok=True)
                    write_ply(os.path.join(args.vis_path, 'test', str(batch) + '_' + str(i) + '_marker.ply'), m)
                    write_mesh(os.path.join(args.vis_path, 'test', str(batch) + '_' + str(i) + '_mesh_gt.ply'), v, face)
                    write_mesh(os.path.join(args.vis_path, 'test', str(batch) + '_' + str(i) + '_mesh.ply'), v_pred, face)
                    write_ply(os.path.join(args.vis_path, 'test', str(batch) + '_' + str(i) + '_joint_gt.ply'), j)
                    write_ply(os.path.join(args.vis_path, 'test', str(batch) + '_' + str(i) + '_joint.ply'), j_pred)

                batch += 1


    print(' - mpjpe: {:6.4f}, mpvpe: {:6.4f}'.format(torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean()))


if __name__ == '__main__':
    gen_data()
    test()
    

