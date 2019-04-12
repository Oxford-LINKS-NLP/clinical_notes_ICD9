"""
    Saving relevant things.
"""
import csv
import json

import math
import numpy as np
import torch

import pandas as pd

top_10_codes = ['401.9', '38.93', '428.0', '427.31', '414.01', '96.04', '96.6', '584.9', '250.00' ,'96.71']

def save_metrics(dicts, metrics_hist, model_dir, metrics_codes=None, metrics_inst=None, hadm_ids=None, test_only=False):
    filename = "/metrics.json" if not test_only else "/metrics_test_only.json"
    with open(model_dir + filename, 'w') as metrics_file:
        #concatenate train, dev and test metrics into one dict
        metrics_hist_train, metrics_hist_dev, metrics_hist_test = metrics_hist
        data = {}
        data.update({"%s_train" % (name):val for (name,val) in metrics_hist_train.items()})
        data.update({"%s_dev" % (name):val for (name,val) in metrics_hist_dev.items()})
        if metrics_hist_test is not None:
            data.update({"%s_test" % (name):val for (name,val) in metrics_hist_test.items()})
        json.dump(data, metrics_file, indent=1)
    if metrics_codes is not None and metrics_inst is not None:
        with open(model_dir + '/metrics_per_code.csv', 'w') as f1, open(model_dir + '/metrics_per_doc.csv', 'w') as f2:
            f1_writer = csv.writer(f1, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            f1_writer.writerow(['code', 'desc', 'valid', 'billable', 'rank', 'train examples', 'test examples', 'accuracy', 'precision', 'recall', 'f1'])
            for row in zip(dicts['ind2c'].values(), dicts['desc_plain'], dicts['valid'].values(), dicts['billable'].values(), dicts['rank'].values(), dicts['train_examples'].values(), dicts['test_examples'].values(), *metrics_codes):
                f1_writer.writerow(row)
            f2_writer = csv.writer(f2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            f2_writer.writerow(['hadm_id', 'accuracy', 'precision', 'recall', 'f1'])
            for row in zip(hadm_ids, *metrics_inst):
                f2_writer.writerow(row)

def save_params_dict(params):
    with open(params["model_dir"] + "/params.json", 'w') as params_file:
        json.dump(params, params_file, indent=1)

def write_docs_top_10(model_dir, fold, df, c2ind):
    for code in top_10_codes:
        ind = c2ind[code]
        
        docs_tp = df[df.apply(lambda x: bool(x['target'][ind] == 1) & bool(x['prediction'][ind] == 1), axis=1)]
        docs_tp['attention'] = docs_tp['attention'].apply(lambda a : a[ind])
        n_samples = 5 if len(docs_tp)>=5 else len(docs_tp)
        if n_samples > 0:
            out_path = model_dir + '/docs_' + code + '_tp_' + fold + '.json'
            with open(out_path, 'w') as f:
                json.dump(json.loads(docs_tp.sample(n=n_samples)[['id', 'text', 'attention']].to_json(orient='records')), f, indent=1)
                
        #docs_tn = df[df.apply(lambda x: bool(x['target'][ind] == 0) & bool(x['prediction'][ind] == 0), axis=1)]
        #docs_tn['attention'] = docs_tn['attention'].apply(lambda a : a[ind])
        #n_samples = 5 if len(docs_tn)>=5 else len(docs_tn)
        #if n_samples > 0:
        #    out_path = model_dir + '/docs_' + code + '_tn_' + fold + '.json'
        #    with open(out_path, 'w') as f:
        #        json.dump(json.loads(docs_tn.sample(n=n_samples)[['id', 'text', 'attention']].to_json(orient='records')), f, indent=1)
        
        docs_fp = df[df.apply(lambda x: bool(x['target'][ind] == 0) & bool(x['prediction'][ind] == 1), axis=1)]
        docs_fp['attention'] = docs_fp['attention'].apply(lambda a : a[ind])
        n_samples = 5 if len(docs_fp)>=5 else len(docs_fp)
        if n_samples > 0:
            out_path = model_dir + '/docs_' + code + '_fp_' + fold + '.json'
            with open(out_path, 'w') as f:
                json.dump(json.loads(docs_fp.sample(n=n_samples)[['id', 'text', 'attention']].to_json(orient='records')), f, indent=1)
        
        docs_fn = df[df.apply(lambda x: bool(x['target'][ind] == 1) & bool(x['prediction'][ind] == 0), axis=1)]
        docs_fn['attention'] = docs_fn['attention'].apply(lambda a : a[ind])
        n_samples = 5 if len(docs_fn)>=5 else len(docs_fn)
        if n_samples > 0:
            out_path = model_dir + '/docs_' + code + '_fn_' + fold + '.json'
            with open(out_path, 'w') as f:
                json.dump(json.loads(docs_fn.sample(n=n_samples)[['id', 'text', 'attention']].to_json(orient='records')), f, indent=1)
        
def write_docs(model_dir, fold, df, lower, upper):
    assert upper >= lower
    
    docs = df[(df['f1']>=lower) & (df['f1']<=upper)]
    n_samples = 5 if len(docs)>=5 else len(docs)
    docs = docs.sample(n=n_samples)
    
    if len(docs) < 1:
        return
            
    i = 0
    for _, doc in docs.iterrows():
        i += 1
        #doc_tp = doc[doc.apply(lambda x: bool(x['target'][ind] == 1) & bool(x['prediction'][ind] == 1), axis=1)]
        tp = np.nonzero((doc['target'] == 1) & (doc['prediction'] == 1))[0]
        #tn = [int(c) for c in (doc['target'] == 0) & (doc['prediction'] == 0) if c]
        fp = [int(c) for c in (doc['target'] == 0) & (doc['prediction'] == 1) if c]
        fn = [int(c) for c in (doc['target'] == 1) & (doc['prediction'] == 0) if c]
            
        doc_tp = pd.DataFrame()
        doc_tp['id'] = [n for n in range(len(tp))]
        doc_tp['text'] = [doc['text'] for n in range(len(tp))]
        doc_tp['attention'] = [a for a in doc['attention'][tp]]
        doc_tp['label'] = [t for t in tp]
        doc_tp['prediction'] = [p for p in tp]
        doc_tp['posterior'] = [0 for p in tp]
        
        for j in range(len(doc_tp)):
            pad = int((len(doc_tp['attention'][j]) - len(doc_tp['text'][j]))/2)
            doc_tp['attention'][j] = doc_tp['attention'][j][pad:-pad] if pad > 0 else doc_tp['attention'][j]

        out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_tp_' + fold + '.json'
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_tp.to_json(orient='records')), f, indent=1)
            
        #doc_tn = pd.DataFrame()
        #doc_tn['id'] = [n for n in range(len(tn))]
        #doc_tn['text'] = [doc['text'] for n in range(len(tn))]
        #doc_tn['attention'] = [a for a in doc['attention'][tn]]
        #doc_tn['label'] = [t for t in tn]
        #doc_tn['prediction'] = [p for p in tn]
        #doc_tn['posterior'] = [0 for p in tn]
               
        #out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_tn_' + fold + '.json'
        #with open(out_path, 'w') as f:
        #    json.dump(json.loads(doc_tn.to_json(orient='records')), f, indent=1)
            
        doc_fp = pd.DataFrame()
        doc_fp['id'] = [n for n in range(len(fp))]
        doc_fp['text'] = [doc['text'] for n in range(len(fp))]
        doc_fp['attention'] = [a for a in doc['attention'][fp,:]]
        doc_fp['label'] = [t for t in fp]
        doc_fp['prediction'] = [p for p in fp]
        doc_fp['posterior'] = [0 for p in fp]
        
        out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_fp_' + fold + '.json'
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_fp.to_json(orient='records')), f, indent=1)
            
        doc_fn = pd.DataFrame()
        doc_fn['id'] = [n for n in range(len(fn))]
        doc_fn['text'] = [doc['text'] for n in range(len(fn))]
        doc_fn['attention'] = [a for a in doc['attention'][fn]]
        doc_fn['label'] = [t for t in fn]
        doc_fn['prediction'] = [p for p in fn]
        doc_fn['posterior'] = [0 for p in fn]
                
        out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_fn_' + fold + '.json'
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_fn.to_json(orient='records')), f, indent=1)

def write_preds(hids, docs, attns, ys, yhats, yhats_raw, metrics, model_dir, fold, ind2c, c2ind, desc_plain):
        
    preds_file = "%s/preds_%s.psv" % (model_dir, fold)
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for yhat, hid in zip(yhats, hids):
            codes = sorted([ind2c[ind] for ind in np.nonzero(yhat)[0]])
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
                
    preds_file_target = "%s/preds_%s_target.psv" % (model_dir, fold)
    with open(preds_file_target, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for y, hid in zip(ys, hids):
            codes = sorted([ind2c[ind] for ind in np.nonzero(y)[0]])
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
                
    labels = {str(k):{'name':'{} {}'.format(ind2c[k], v) } for k, v in enumerate(desc_plain)}
    with open(model_dir + '/labels.json', 'w') as f:
        json.dump(labels, f, indent=1)
    
    metrics_df = pd.DataFrame(data={'id':hids, 'accuracy':metrics[0], 'precision':metrics[1], 'recall':metrics[2], 'f1':metrics[3]})
    #docs_df = pd.DataFrame(data={'id':hids, 'text':docs, 'attention': attns, 'target':ys, 'prediction':yhats})
    docs_df = pd.DataFrame()
    docs_df['id'] = hids
    docs_df['text'] = docs
    docs_df['target'] = [y for y in ys]
    docs_df['prediction'] = [yhat for yhat in yhats]
    docs_df['attention'] = attns
    
    merged_df = docs_df.merge(metrics_df, on='id', how='inner')
    
    write_docs_top_10(model_dir, fold, merged_df, c2ind)
    
    write_docs(model_dir, fold, merged_df, 0.9, 1.0)
    write_docs(model_dir, fold, merged_df, 0.7, 0.8)
    write_docs(model_dir, fold, merged_df, 0.5, 0.6)
    write_docs(model_dir, fold, merged_df, 0.3, 0.4)
    write_docs(model_dir, fold, merged_df, 0.1, 0.2)
    
    #output = []
    #for hid, doc, attn in zip(hids, docs, attns):
    #    m = min(len(doc), len(attn))
    #    output.append({'id' : str(hid), 'text' : doc[:m], 'attention' : attn[:m]})
           
    #out_path = model_dir + '/output_' + fold + '.json'
    #with open(out_path, 'w') as f:
    #    json.dump(output, f, indent=1)

def save_everything(args, dicts, metrics_hist_all, model, model_dir, params, criterion, metrics_codes=None, metrics_inst=None, hadm_ids=None, evaluate=False, test_only=False):
    """
        Save metrics, model, params all in model_dir
    """
    save_metrics(dicts, metrics_hist_all, model_dir, metrics_codes=metrics_codes, metrics_inst=metrics_inst, hadm_ids=hadm_ids, test_only=test_only)
    params['model_dir'] = model_dir
    save_params_dict(params)

    if not evaluate and not test_only:
        sd = model.cpu().state_dict()
        torch.save(sd, model_dir + "/model_last_epoch.pth")

        #save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[1][criterion])):
            if np.nanargmax(metrics_hist_all[1][criterion]) == len(metrics_hist_all[1][criterion]) - 1:
                #save state dict
                sd = model.state_dict()
                torch.save(sd, model_dir + "/model_best_%s.pth" % criterion)
        if args.gpu:
            model.cuda()
    print("saved metrics, params, model to directory %s\n" % (model_dir))
