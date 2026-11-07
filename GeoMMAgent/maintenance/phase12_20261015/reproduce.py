"""Rebuild the exact source pack and replay into a new D experiment directory."""
import argparse
import sys
from pathlib import Path
import numpy as np
from run_experiment import ROOT,OUT,read,write,sha,array_sha,normalize,now


def rebuild_pack(destination):
    source=read(OUT/'protocol.json')
    train_x,train_y,train_groups,val_x,val_y,val_groups=[],[],[],[],[],[]
    for group,region in enumerate(source['provenance']):
        for split in ['train','val']:
            row=region['splits'][split]
            raw=np.array(np.load(row['input_path'],mmap_mode='r')[row['indices']],dtype=np.float32)
            labels=np.array(np.load(row['label_path'],mmap_mode='r')[row['indices']],dtype=np.float32).reshape(-1,1,32,32)
            if array_sha(raw)!=row['selected_raw_sha256'] or array_sha(labels)!=row['selected_labels_sha256']:
                raise ValueError('Selected source pixels or labels changed; do not silently rebuild')
            x=normalize(raw)
            if split=='train':
                train_x.append(x);train_y.append(labels);train_groups.append(np.full(len(x),group,np.int64))
            else:
                val_x.append(x);val_y.append(labels);val_groups.extend([group]*len(x))
    destination=Path(destination)
    np.savez_compressed(destination,train_x=np.concatenate(train_x),train_y=np.concatenate(train_y),
        train_group=np.concatenate(train_groups),val_x=np.concatenate(val_x),val_y=np.concatenate(val_y),val_group=np.array(val_groups))
    if sha(destination)!=source['pack_sha256']:
        raise ValueError('Rebuilt source pack differs from frozen experiment')
    return source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['pack','latent','feature','legacy','all'],default='pack')
    args=parser.parse_args()
    target=Path(args.output).resolve()
    if target.drive.upper()!='D:' or not target.is_relative_to((ROOT/'experiments').resolve()) or target.exists():
        raise ValueError('Use a new directory strictly inside D:/LMHLD/experiments')
    target.mkdir(parents=True)
    parent=rebuild_pack(target/'source_pack.npz')
    protocol={**parent,'pack_path':str(target/'source_pack.npz'),'reproduction_of':str(OUT),
              'reproduction_registered_at':now(),'historical_records_and_retired_weights_not_overwritten':True}
    write(target/'protocol.json',protocol)
    print('Exact source pack reconstructed:',sha(target/'source_pack.npz'),flush=True)
    if args.stage=='pack':return
    import run_experiment as runner
    runner.OUT=target;runner.HERE=target
    if args.stage in ['latent','all']:runner.run()
    # Follow-ups need the same mine-evaluation provenance, not new selection.
    write(target/'mine_evaluation_plan.json',read(OUT/'mine_evaluation_plan.json'))
    if args.stage in ['feature','all']:
        import run_feature_residual as feature
        feature.OUT=target;feature.HERE=target;feature.FOLLOW=target/'feature_residual'
        feature.main()
    if args.stage in ['legacy','all']:
        import refine_legacy as legacy
        legacy.OUT=target;legacy.HERE=target;legacy.FOLLOW=target/'legacy_task_refinement'
        legacy.main()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
