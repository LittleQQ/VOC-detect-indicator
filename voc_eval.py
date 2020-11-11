# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import numpy as np
import json 
from globalpublicfuns import *


class VOCDetIndicator:
    def __init__(self,testsetdirs,detresulttxtdir,databasename,clsnames,mergeclsnamesdict,iouthresh=0.5,use_07_metric=False):
        self.testsetdirs=testsetdirs
        self.clsnames=clsnames
        self.mergeclsnamesdict=mergeclsnamesdict
        self.detresulttxtdir=detresulttxtdir
        self.iouthresh=iouthresh
        self.use_07_metric=use_07_metric

        def getannotgroundtruth():
            def parse_rec(filename):
                """ Parse a PASCAL VOC xml file """
                tree = ET.parse(filename)
                objects = []
                for obj in tree.findall('object'):
                    obj_struct = {}
                    cls0 = obj.find('name').text
                    if cls0 in self.mergeclsnamesdict.keys():
                        print(filename)
                    obj_struct['name'] = cls0
                    # obj_struct['pose'] = obj.find('pose').text
                    obj_struct['truncated'] = int(obj.find('truncated').text)
                    obj_struct['difficult'] = int(obj.find('difficult').text)
                    bbox = obj.find('bndbox')
                    obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                                          int(float(bbox.find('ymin').text)),
                                          int(float(bbox.find('xmax').text)),
                                          int(float(bbox.find('ymax').text))]
                    objects.append(obj_struct)
                # print 'voc_eval.py-->parse_rec()-->objects='+str(objects)+'\n'
                return objects

            recs = {}
            annotcachefilepath = pathjoin(detresulttxtdir, databasename + '_annot.json')
            if os.path.exists(annotcachefilepath):
                print('reading annotation cache file')
                with open(annotcachefilepath, 'r') as f:
                    recs = json.load(f)
            else:
                i = 0
                for testsetdir in testsetdirs:
                    annotpaths = getxextnamefilefromdir(testsetdir, fileextname='.xml')
                    for annotpath in annotpaths:
                        filename = os.path.basename(annotpath)[:-4]
                        recs[filename] = parse_rec(annotpath)
                        i += 1
                        if i % 500 == 0:
                            print('Reading annotation for {:d}'.format(i))
                # save

                print('Saving cached annotations to {:s}'.format(annotcachefilepath))
                with open(annotcachefilepath, 'w') as fp:
                    json.dump(recs, fp,indent=4)

            return recs

        self.annotgroundtruth=getannotgroundtruth()



    def voc_ap(self,rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            #
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self,clsname):
        recs = self.annotgroundtruth
        # extract gt objects for this class
        class_recs = {}
        npos = 0  #
        for imagename in recs.keys():
            R = [obj for obj in recs[imagename] if obj['name'] == clsname]

            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        detresulttxtname = '{}.txt'.format(clsname)
        detfile =pathjoin(self.detresulttxtdir,detresulttxtname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)  #
        tp = np.zeros(nd)  #
        fp = np.zeros(nd)  #

        for d in range(nd):
            R = class_recs[image_ids[d]]  #
            bb = BB[d, :].astype(float)  #
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  #

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)  #
                jmax = np.argmax(overlaps)

            if ovmax > self.iouthresh:  #
                if not R['difficult'][jmax]:  #
                    if not R['det'][jmax]:  #
                        tp[d] = 1.  #
                        R['det'][jmax] = 1  #
                    else:
                        fp[d] = 1.  #
            else:
                fp[d] = 1.  #


        # compute precision recall
        fp = np.cumsum(fp)  #
        tp = np.cumsum(tp)  #
        rec = tp / float(npos)  #
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  #
        ap = self.voc_ap(rec, prec, self.use_07_metric)  #

        return rec, prec, ap



    def gettestsetdetindicator(self):
        precisions,recalls,aps={},{},{}
        for clsname in self.clsnames:
            rec, prec, ap = self.voc_eval(clsname)

            precisions[clsname]=round(prec[-1],3)
            recalls[clsname]=round(rec[-1],3)
            aps[clsname]=round(ap,3)

        map=0
        for cls in aps.keys():
            map+=aps[cls]
        map=map/len(aps)
        map=round(map,3)

        precision_recall_ap_map = {'precision':precisions,'recall':recalls,'ap':aps,'map':map}

        return precision_recall_ap_map





