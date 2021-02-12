import pandas as pd
import numpy as np
import os
import functools
import glob
import csv
from collections import namedtuple
import re
import tqdm
import SimpleITK as sitk
from torch.utils.data import Dataset
import copy
import torch
import matplotlib.pyplot as plt

annotations = pd.read_csv('annotations.csv')
candidates = pd.read_csv('candidates.csv')


@functools.lru_cache(1)
def create_candidate_frame(on_disk = True):

  all_on_disk = glob.glob('subset*/*.mhd')
  only_names = [re.search(r'subset\d/(.*).mhd', x).group(1) for x in all_on_disk]

  annotations_frame = pd.read_csv('annotations.csv')
  annotations_frame = annotations_frame[annotations_frame['seriesuid'].isin(only_names)]
  annotations_frame = annotations_frame.reset_index()

  diameters_dict = {}
  for name in only_names:
    diameters_dict[name] = [row.to_list() for i, row in annotations_frame[annotations_frame['seriesuid'] == name].iloc[:, -4:].iterrows()]
    

  candidates_frame = pd.read_csv('candidates.csv')
  candidates_frame = candidates_frame[candidates_frame['seriesuid'].isin(only_names)]
  candidates_frame['diameter_mm'] = [0.0 for i in range(len(candidates_frame))]

  with open('candidates.csv', "r") as f:
      for index, row in tqdm.tqdm(enumerate(list(csv.reader(f))[1:])):
          
          uid = row[0]   
          if uid not in diameters_dict.keys():
              continue

          coords = [float(x) for x in row[1:4]]
          class_ = row[4]

          candidateDiameter_mm = 0.0
          for annotation_tup in diameters_dict.get(uid, []):
              annotationCenter_xyz = annotation_tup
              for i in range(3):
                  delta_mm = abs(coords[i] - annotation_tup[i])
                  if delta_mm > annotation_tup[3] / 4:
                      break
              else:
                  candidates_frame.loc[index, 'diameter_mm'] = annotation_tup[3]
                  break

  return  candidates_frame.sort_values('diameter_mm', ascending=False)


IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    print('2')
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
            ):
        self.candidates_frame = copy.copy(candidates_frame).reset_index()

        if series_uid:
            self.candidates_frame = self.candidates_frame[self.candidates_frame['seriesuid'] == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidates_frame = self.candidates_frame.iloc[::val_stride,:]
            assert not self.candidates_frame.empty
        elif val_stride > 0:
            self.candidates_frame.drop(self.candidates_frame.index[::val_stride])
            assert not self.candidates_frame.empty

        print("{!r}: {} {} samples".format(
            self,
            len(self.candidates_frame),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        item = self.candidates_frame.iloc[ndx,:]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            item[1],
            tuple(item[2:5]),
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not item[5],
                item[5]
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            item[1],
            torch.tensor(center_irc),
        )





def showCandidate(series_uid, batch_ndx=None, **kwargs):
   
    clim=(-1000.0, 300)
   
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    pos_list = ds.candidates_frame[ds.candidates_frame['class'] == 1].index.to_list()
    if batch_ndx is None:
        if pos_list:
            batch_ndx = 0
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:,int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:,:,int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,ct_a.shape[1]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,:,ct_a.shape[2]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')


    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)