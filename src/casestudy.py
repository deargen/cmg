import os
import tqdm
import math
import time
import argparse
import numpy as np
import tensorflow as tf
import _pickle as cPickle
from itertools import repeat
from multiprocessing import Pool
from src.official.transformer.v2 import optgen_v8, optgen_v9, optgen_v11, optgen_v12, optgen_v13, optgen_v21
from src.score_result import penalized_logp, qed, drd2, similarity
from src.official.transformer.utils.molecule_tokenizer import Moltokenizer
from src.official.transformer.v2 import optgen_v1, optgen_v2, optgen_v3, optgen_v4, optgen_v5, optgen_v6, optgen_v7


__author__ = 'Bonggun Shin'


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("model", None)
def get_dev_model(params):
    if get_dev_model.model is None:
        # print("dev_model is created!!!")
        model_name = "optgen_%s" % params["model_version"]
        get_dev_model.model = eval(model_name).create_model(params, is_train=False)
        print("=========================== %s dev created!! ==========================" % model_name)

    else:
        print("dev_model is reused!!!")

    return get_dev_model.model


def get_layer_index(model, name=None, custom_name=None):
    for idx, l in enumerate(model.layers):
        if name=="embedding" and "embedding_freezable" in l.name:
            if custom_name==l.custom_name:
                return idx
        else:
            if name==l.name:
                return idx

    print("wrong name", name)
    exit()
    return -1


def transfer_weights_from_propnet(model, params):
    propnet_weight_path = params["propnet_weight_path"]
    propnet_weights = cPickle.load(open(propnet_weight_path, 'rb'))

    model_layers = [l for l in model.layers]
    # model_layers[63]  # Emb 42
    # model_layers[64]  # Bidirectional 43
    # model_layers[65]  # idense
    # model_layers[66]  # Dense2 44
    layer_index = get_layer_index(model, name='embedding', custom_name='propnet_emb')
    print("setting propnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights([propnet_weights["embedding_shared_weights"]])

    layer_index = get_layer_index(model, name='propnet_bidirectional')
    print("setting propnet weights: %s..." % model_layers[layer_index].name)
    lstm_weights = []
    lstm_weights.append(propnet_weights["forward_prop_lstm_0"])
    lstm_weights.append(propnet_weights["forward_prop_lstm_1"])
    lstm_weights.append(propnet_weights["forward_prop_lstm_2"])
    lstm_weights.append(propnet_weights["backward_prop_lstm_0"])
    lstm_weights.append(propnet_weights["backward_prop_lstm_1"])
    lstm_weights.append(propnet_weights["backward_prop_lstm_2"])
    model_layers[layer_index].set_weights(lstm_weights)

    layer_index = get_layer_index(model, name='propnet_idense')
    output_weights = []
    output_weights.append(propnet_weights["idense0"])
    output_weights.append(propnet_weights["idense1"])
    print("setting propnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights(output_weights)

    layer_index = get_layer_index(model, name='propnet_output')
    output_weights = []
    output_weights.append(propnet_weights["output0"])
    output_weights.append(propnet_weights["output1"])
    print("setting propnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights(output_weights)

    print("setting propnet weights done!!!")


def transfer_weights_from_simnet(model, params):
    simnet_weight_path = params["simnet_weight_path"]
    simnet_weights = cPickle.load(open(simnet_weight_path, 'rb'))
    model_layers = [l for l in model.layers]
    # model_layers[63]  # Emb 42
    # model_layers[64]  # Bidirectional 43
    # model_layers[65]  # idense
    # model_layers[66]  # Dense2 44
    layer_index = get_layer_index(model, name='embedding', custom_name='simnet_emb')
    print("setting simnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights([simnet_weights["embedding_shared_weights"]])

    layer_index = get_layer_index(model, name='simnet_bidirectional')
    print("setting simnet weights: %s..." % model_layers[layer_index].name)
    lstm_weights = []
    lstm_weights.append(simnet_weights["forward_sim_lstm_0"])
    lstm_weights.append(simnet_weights["forward_sim_lstm_1"])
    lstm_weights.append(simnet_weights["forward_sim_lstm_2"])
    lstm_weights.append(simnet_weights["backward_sim_lstm_0"])
    lstm_weights.append(simnet_weights["backward_sim_lstm_1"])
    lstm_weights.append(simnet_weights["backward_sim_lstm_2"])
    model_layers[layer_index].set_weights(lstm_weights)

    layer_index = get_layer_index(model, name='simnet_idense')
    output_weights = []
    output_weights.append(simnet_weights["simdense0"])
    output_weights.append(simnet_weights["simdense1"])
    print("setting simnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights(output_weights)

    layer_index = get_layer_index(model, name='simnet_output')
    output_weights = []
    output_weights.append(simnet_weights["simoutput0"])
    output_weights.append(simnet_weights["simoutput1"])
    print("setting simnet weights: %s..." % model_layers[layer_index].name)
    model_layers[layer_index].set_weights(output_weights)

    print("setting simnet weights done!!!")


@static_var("model", None)
def get_trn_model(params):
    if get_trn_model.model is None:
        print("trn_model is creating!!!")
        model_name = "optgen_%s" % params["model_version"]
        get_trn_model.model = eval(model_name).create_model(params, is_train=True)
        print("=========================== %s trn created!! ==========================" % model_name)

    else:
        print("trn_model is reused!!!")

    return get_trn_model.model


def get_model(params, epoch_num):

    trn_model = get_trn_model(params)
    model_filename = "%s/%s" % (params["model_dir"], "cp-%04d.ckpt" % (epoch_num))
    print("Load weights: {}".format(model_filename))
    trn_model.load_weights(model_filename).expect_partial()

    dev_model = get_dev_model(params)
    dev_model.summary()

    # model.layers[12].set_weights(trn_model.layers[4].get_weights())
    model_name = "optgen_%s" % params["model_version"]
    trn_optgen_layer_index = get_layer_index(trn_model, name=model_name)
    dev_optgen_layer_index = get_layer_index(dev_model, name=model_name)
    dev_model.layers[dev_optgen_layer_index].set_weights(trn_model.layers[trn_optgen_layer_index].get_weights())
    print("trn_optgen_layer_index(%d, %s), dev_optgen_layer_index(%d, %s)" %
          (trn_optgen_layer_index, trn_model.layers[trn_optgen_layer_index].name,
           dev_optgen_layer_index, dev_model.layers[dev_optgen_layer_index].name))
    print("set weights for (%s) layer" % dev_model.layers[dev_optgen_layer_index].name)

    if params["use_propnet"] == 1:
        print("========================== Transferring PropNET weights....==========================")
        transfer_weights_from_propnet(dev_model, params)
    if params["use_simnet"] == 1:
        print("========================== Transferring SimNET weights....==========================")
        transfer_weights_from_simnet(dev_model, params)

    return dev_model


def get_score(src, trg, task='logp04'):
    if task=='logp04':
        logp_improvement = penalized_logp(trg) - penalized_logp(src)
        # return logp_improvement * (0.6+min(sim, 0.4))
        return logp_improvement

    elif task=='qed':
        return qed(trg)

    elif task=='drd2':
        try:
            val = drd2(trg)
        except:
            print("***************\n***************\n***************\n***************\n")
            print("***************\n***************\n***************\n***************\n")
            print(trg)
            print("***************\n***************\n***************\n***************\n")
            print("***************\n***************\n***************\n***************\n")
            val = 0
        return val

    else:
        assert 'wrong task: %s' % task



def worker_wrapper(args):
    return worker(*args)

def worker(outputs, smiles_list, n, wid, start_index, end_index):
    """
    Args:
      return_dic: store result here
      n: int, number of data
      wid: int, worker id
      strat_index: int, the start index of original data that this worker will work on
      end_index: int, the end index of original data that this worker will work on
    """
    print('[worker-%d] start_index(%d), end_index(%d) n(%d)' %
          (wid, start_index, end_index, n))

    if end_index > n:
        end_index = n

    score_props = []
    valid_smiles_list=[]
    for idx in range(start_index, end_index, 1):
        output = outputs[idx]
        smiles_x = smiles_list[idx]
        smiles_y = moltokenizer.decode(output)
        sim = similarity(smiles_x, smiles_y)
        if sim < 0.4:
            score_props.append([0., 0.,0.])
            valid_smiles_list.append('')
        else:
            score_plogp = get_score(smiles_x, smiles_y, task='logp04')
            score_qed = get_score(smiles_x, smiles_y, task='qed')
            score_drd2 = get_score(smiles_x, smiles_y, task='drd2')
            score_props.append([score_plogp, score_qed, score_drd2])
            valid_smiles_list.append(smiles_y)

    return score_props, valid_smiles_list


def evaluate(test_smiles_list, model, moltokenizer, log_diff_list, qed_diff_list, drd2_diff_list):
    smiles_list = []
    ids_list = []
    property_x_list = []
    property_desired_list = []

    n_grid = 3 * 3 * 1

    for idx, smiles in enumerate(test_smiles_list):
        ids = moltokenizer.encode(smiles)
        logp_val = penalized_logp(smiles)
        qed_val = qed(smiles)
        drd2_val = drd2(smiles)

        property_x = np.expand_dims(np.array([logp_val, qed_val, drd2_val]), axis=0)

        for log_diff in log_diff_list:
            logp_improvement = logp_val+log_diff
            for qed_diff in qed_diff_list:
                qed_desired = qed_val+qed_diff
                for drd2_diff in drd2_diff_list:

                    drd2_desired = drd2_val+drd2_diff

                    property_desired = np.expand_dims(
                        np.array([logp_improvement, qed_desired, drd2_desired]), axis=0)

                    smiles_list.append(smiles)
                    ids_list.append(ids)
                    property_x_list.append(property_x)
                    property_desired_list.append(property_desired)

    x = np.array(tf.keras.preprocessing.sequence.pad_sequences(ids_list, dtype="int64", padding="post"))
    px = np.concatenate(property_x_list, axis=0)
    py = np.concatenate(property_desired_list, axis=0)

    print(px[2,:])
    print(py[2,:])

    print(px[10, :])
    print(py[10, :])

    outputs, _ = model.predict([x, px, py], batch_size=args.eb, verbose=1)

    if args.p == 1:
        valid_smiles_list = []
        score_plogp_all = []
        score_qed_all = []
        score_drd2_all = []
        for idx, output in tqdm.tqdm(enumerate(outputs)):
            smiles_x = smiles_list[idx]
            smiles_y = moltokenizer.decode(output)
            sim = similarity(smiles_x, smiles_y)
            if sim < 0.4:
                score_plogp_all.append(0)
                score_qed_all.append(0)
                score_drd2_all.append(0)
                valid_smiles_list.append('')
            else:
                score_plogp = get_score(smiles_x, smiles_y, task='logp04')
                score_qed = get_score(smiles_x, smiles_y, task='qed')
                score_drd2 = get_score(smiles_x, smiles_y, task='drd2')
                score_plogp_all.append(score_plogp)
                score_qed_all.append(score_qed)
                score_drd2_all.append(score_drd2)
                valid_smiles_list.append(smiles_y)

    else:
        with Timer("score multi calculation..."):
            n = len(outputs)
            n_proc = args.p
            batch = math.ceil(n / (n_proc))

            with Pool(processes=n_proc) as pool:
                r = pool.map_async(worker_wrapper,
                                   zip(repeat(outputs), repeat(smiles_list),
                                       repeat(n), range(1, n_proc + 1), range(0, n, batch),
                                       range(batch, batch * n_proc + 1, batch))
                                   )
                r.wait()

            score_all = []
            valid_smiles_list = []
            for partial_score, valid_smiles_p_list in r.get():
                score_all+=partial_score
                valid_smiles_list+=valid_smiles_p_list

            score_all = np.array(score_all)
            score_plogp_all = score_all[:,0]
            score_qed_all = score_all[:,1]
            score_drd2_all = score_all[:,2]

    return score_plogp_all, score_qed_all, score_drd2_all, valid_smiles_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', type=str, default="../../data", help='base path.')

    parser.add_argument('-g', default="3", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str)
    parser.add_argument('-lr', default="2", type=str) # 1e-3
    parser.add_argument('-lp', default="1.0", type=float)  # 1e-3
    parser.add_argument('-he', default=8, type=int, help='num_heads')
    parser.add_argument('-hi', default=4, type=int, help='num_hidden_layers')
    parser.add_argument('-f', default=256, type=int, help='filter_size')
    parser.add_argument('-b', default=4096, type=int, help='batch_size') # 4096, 2048
    parser.add_argument('-v', default=21, type=int, help='model version')
    parser.add_argument('-t', default=2, type=int, help='attempt')
    parser.add_argument('-es', default=500, type=int, help='eval epoch start')
    parser.add_argument('-ee', default=501, type=int, help='eval epoch end')
    parser.add_argument('-et', default=10, type=int, choices=[1, 5, 10, 20], help='eval epoch step')
    parser.add_argument('-pnet', default=0, type=int, help='if propnet used')
    parser.add_argument('-snet', default=1, type=int, help='if simnet used')
    parser.add_argument('-bf', default=1, type=int, help='if use_beam_filter')
    parser.add_argument('-p', default=32, type=int, help='number of processes')
    parser.add_argument('-eb', default=2000, type=int, help='evaluate batch')
    parser.add_argument('-data', default=3, type=int, help='dataset kind (0-800r,1-800h,2-50r,3-50h,4-1188,5-80,6-full), 7-j.o')
    parser.add_argument('-rs', default=0, type=int, help='random seed')
    parser.add_argument('-ns', type=int, default=50, help='n_sample.')

    args, unparsed = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model_version = "v%d" % args.v

    base_path = args.base

    n_samples = 10827615
    # dev_smiles_filename = "%s/zinc_drd2_filtered_dev_smiles_dic.cpkl" % base_path

    data_dir = "%s/v10_trn" % base_path

    iterations = 100
    steps_between_evals = n_samples // args.b
    train_steps = steps_between_evals * iterations
    validation_steps = 32
    batch_size = args.b

    model_dir = "%s/model.vv%d.vnet%d.pnet%d.snet%d.head%d.hid%d.fil%d.batch%d.lr%s.t%d" % (base_path, args.v,
                                                                                            args.vnet, args.pnet,
                                                                                            args.snet, args.he, args.hi,
                                                                                            args.f, args.b, args.lr,
                                                                                            args.t)

    mt_weight_path = "%s/mt_weights.cpkl" % base_path

    propnet_weight_path = "%s/propnet.weights.cpkl" % base_path
    simnet_weight_path = "%s/simnet.weights.cpkl" % base_path

    optgen_config_file = "%s/../config/optgen_config.json" % base_path


    params = optgen_v2.load_config(optgen_config_file)
    params["propnet_weight_path"] = propnet_weight_path
    params["simnet_weight_path"] = simnet_weight_path

    params["steps_between_evals"] = steps_between_evals
    params["batch_size"] = batch_size
    params["train_steps"] = train_steps
    params["validation_steps"] = validation_steps
    params["model_dir"] = model_dir
    params["mt_weight_path"] = mt_weight_path
    params["num_heads"] = args.he
    params["num_hidden_layers"] = args.hi
    params["filter_size"] = args.f
    params["learning_rate"] = float(args.lr)
    params["model_version"] = model_version

    params["n_samples"] = n_samples
    params["iterations"] = iterations
    params["data_dir"] = data_dir

    params["vocab_file"] = "%s/optgen_vocab.txt" % base_path
    params["vocab_size"] = 71

    params["use_propnet"] = args.pnet
    params["use_simnet"] = args.snet
    params["use_beam_filter"] = args.bf

    print("========================================[params]========================================")
    for k, v in params.items():
        print(k, ":", v)
    print("========================================[params]========================================")

    moltokenizer = Moltokenizer(params["vocab_file"])
    epoch_num = 270 # drd2 test 0.61, V0011
    # epoch_num = 210
    tst_model = get_model(params, epoch_num)

    selected_smiles='COC1=CC=C(C=C1)C(=O)N1CCCC1=O'

    log_diff_list = [0.0]
    qed_diff_list = [-0.1, -0.05, 0.0, 0.05, 0.1]
    drd2_diff_list = np.array(range(6, 10)) / 10.0

    org_plp = penalized_logp(selected_smiles)
    org_qed = qed(selected_smiles)
    org_drd2 = drd2(selected_smiles)
    score_plogp_all, score_qed_all, score_drd2_all, valid_smiles_list = \
        evaluate([selected_smiles], tst_model, moltokenizer, log_diff_list, qed_diff_list, drd2_diff_list)

    new_plp = score_plogp_all[np.argmax(score_drd2_all)]
    new_qed = score_qed_all[np.argmax(score_drd2_all)]
    new_drd2 = score_drd2_all[np.argmax(score_drd2_all)]
    new_smiles = valid_smiles_list[np.argmax(score_drd2_all)]

    print(org_plp, org_qed, org_drd2)
    print(new_plp, new_qed, new_drd2)

    print("improvemend:", new_drd2 - org_drd2)
    print("diff:", (new_plp - org_plp) ** 2 + (new_qed - org_qed) ** 2)

    print(new_smiles)

    index_list = []
    for i in log_diff_list:
        for j in qed_diff_list:
            for k in drd2_diff_list:
                index_list.append((i,j,k))
    print(index_list[np.argmax(score_drd2_all)])

    print(np.array(index_list)[np.where(score_drd2_all>0.5)])
