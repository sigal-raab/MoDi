import torch
from t2m.networks.modules import *
# from t2m.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
# from t2m.data.dataset import collate_fn
from dataclasses import dataclass
from utils.pre_run import load_all_form_checkpoint
# from generate import sample
import random

# from utils.visualization import motion2humanml
# from utils.data import motion_from_raw
# from generate import get_gen_mot_np
from evaluate import generate
from utils.humanml_utils import position_to_humanml
# modi genrate for eval
#from evaluate import generate
genarate = ()

import os

def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

#
# class CompV6GeneratedDataset(Dataset):
#
#     def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
#         assert mm_num_samples < len(dataset)
#         print(opt.model_dir)
#
#         dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
#         text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
#         trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
#         epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
#         generated_motion = []
#         mm_generated_motions = []
#         mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
#         mm_idxs = np.sort(mm_idxs)
#         min_mov_length = 10 if opt.dataset_name == 't2m' else 6
#         # print(mm_idxs)
#
#         print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
#         trainer.eval_mode()
#         trainer.to(opt.device)
#         with torch.no_grad():
#             for i, data in tqdm(enumerate(dataloader)):
#                 word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
#                 tokens = tokens[0].split('_')
#                 # print(tokens)
#                 word_emb = word_emb.detach().to(opt.device).float()
#                 pos_ohot = pos_ohot.detach().to(opt.device).float()
#
#                 # print(cap_lens)
#                 pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
#                 pred_dis = nn.Softmax(-1)(pred_dis).squeeze()
#
#                 mm_num_now = len(mm_generated_motions)
#                 is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
#                 # if is_mm:
#                 #     print(mm_num_now, i, mm_idxs[mm_num_now])
#                 repeat_times = mm_num_repeats if is_mm else 1
#                 mm_motions = []
#                 # print(m_lens[0].item(), cap_lens[0].item())
#                 for t in range(repeat_times):
#                     mov_length = torch.multinomial(pred_dis, 1, replacement=True)
#                     if mov_length < min_mov_length:
#                         mov_length = torch.multinomial(pred_dis, 1, replacement=True)
#                     if mov_length < min_mov_length:
#                         mov_length = torch.multinomial(pred_dis, 1, replacement=True)
#
#                     m_lens = mov_length * opt.unit_length
#                     pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
#                                                           m_lens[0]//opt.unit_length, opt.dim_pose)
#                     if t == 0:
#                         # print(m_lens)
#                         # print(text_data)
#                         sub_dict = {'motion': pred_motions[0].cpu().numpy(),
#                                     'length': m_lens[0].item(),
#                                     'cap_len': cap_lens[0].item(),
#                                     'caption': caption[0],
#                                     'tokens': tokens}
#                         generated_motion.append(sub_dict)
#
#                     if is_mm:
#                         mm_motions.append({
#                             'motion': pred_motions[0].cpu().numpy(),
#                             'length': m_lens[0].item()
#                         })
#                 if is_mm:
#                     mm_generated_motions.append({'caption': caption[0],
#                                                  'tokens': tokens,
#                                                  'cap_len': cap_lens[0].item(),
#                                                  'mm_motions': mm_motions})
#
#                     # if len(mm_generated_motions) < mm_num_samples:
#                     #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
#         self.generated_motion = generated_motion
#         self.mm_generated_motion = mm_generated_motions
#         # print(len(generated_motion))
#         # print(len(mm_generated_motions))
#         self.opt = opt
#         self.w_vectorizer = w_vectorizer
#
#
#     def __len__(self):
#         return len(self.generated_motion)
#
#
#     def __getitem__(self, item):
#         data = self.generated_motion[item]
#         motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
#         sent_len = data['cap_len']
#         # tokens = text_data['tokens']
#         # if len(tokens) < self.opt.max_text_len:
#         #     # pad with "unk"
#         #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#         #     sent_len = len(tokens)
#         #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
#         # else:
#         #     # crop
#         #     tokens = tokens[:self.opt.max_text_len]
#         #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#         #     sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)
#
#         # print(tokens)
#         # print(caption)
#         # print(m_length)
#         # print(self.opt.max_motion_length)
#         if m_length < self.opt.max_motion_length:
#             motion = np.concatenate([motion,
#                                      np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
#                                      ], axis=0)
#         return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
#

class ModiGeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats, args, mean, std):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        # LOAD DATASET
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
        # dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

        # LOAD MODEL FROM CHECKPOINT
        g_ema, discriminator, checkpoint, entity, mean_joints, std_joints = load_all_form_checkpoint(args.ckpt, args)

        with torch.no_grad():
            g_ema.eval()
            mean_latent = g_ema.mean_latent(args.truncation_mean)
            args.truncation = 1

        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, captions, cap_lens, motions, m_lens, tokens = data
                tokens = [token.split('_') for token in tokens]
                # print(tokens)
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                # if is_mm:
                #     print(mm_num_now, i, mm_idxs[mm_num_now])
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = [[]] * len(captions)
                # print(m_lens[0].item(), cap_lens[0].item())
                for t in range(repeat_times):
                    # should be (1 X mov_length X movment)
                    pred_motions, _ = generate(args, g_ema, args.device, mean_joints, std_joints, entity, captions)

                    # # find file
                    # p = '/content/drive/MyDrive/MoDi/MoDi/examples/HumanML_raw'

                    # n = 0
                    # for fn in os.listdir(os.path.join(p,'texts')):
                    #     p0=os.path.join(p,'texts',fn)
                    #     with open(p0, 'r') as f:
                    #         lines = f.readlines()
                    #         for line in lines:
                    #             if line.split('#')[0]==captions[0]:
                    #                 n=fn[:-4]

                    # # load original
                    # pred_motions = [np.load(os.path.join(p,'new_joint_vecs',f'{n}.npy'))]

                    # # load preprocessed and unprocess
                    # from Motion.BVH import load
                    # from Motion.Animation import positions_global
                    # pred_motions = np.array([])

                    # a, nm,_ = load(os.path.join(p,'processed',f'{n}_joints_1_frames_0.bvh'))
                    # pred_motions = positions_global(a)
                    
                    # pred_motions,_,_,_ = position_to_humanml(pred_motions, nm)
                    # pred_motions = [pred_motions]

                    for j in range(len(pred_motions)):
                        if t == 0:
                            # print(m_lens)
                            # print(text_data)
                            sub_dict = {'motion': pred_motions[j],
                                        'length': pred_motions[j].shape[0],
                                        'cap_len': cap_lens[j].item(),
                                        'caption': captions[j],
                                        'tokens': tokens[j]}
                            generated_motion.append(sub_dict)

                        if is_mm:
                            mm_motions[j].append({
                                'motion': pred_motions[j],
                                'length': pred_motions[j].shape[0]
                            })
                for j in range(len(captions)):
                    if is_mm:
                        mm_generated_motions.append({'caption': captions[j],
                                                     'tokens': tokens[j],
                                                     'cap_len': cap_lens[j].item(),
                                                     'mm_motions': mm_motions[j]})

                    # if len(mm_generated_motions) < mm_num_samples:
                    #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # print(len(generated_motion))
        # print(len(mm_generated_motions))
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.mean = mean
        self.std = std



    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        ''' humanml eval preproccess '''
        # Crop the motions in to times of 4, and introduce small variations
        # if self.opt.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std


        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)