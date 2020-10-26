from src.official.transformer.v2 import data_pipeline
from src.official.transformer.v2 import optgen_v1, optgen_v2, optgen_v3, optgen_v4, optgen_v5, optgen_v6, optgen_v7
from src.official.transformer.v2 import optgen_v8, optgen_v9, optgen_v11, optgen_v12, optgen_v13, optgen_v21
from src.official.transformer.v2 import optimizer
from src.official.transformer.utils.molecule_tokenizer import Moltokenizer
import numpy as np
import tensorflow as tf
import os
import _pickle as cPickle
from src.score_result import penalized_logp, qed, drd2, similarity
import tqdm
import argparse
import sys


__author__ = 'Bonggun Shin'


def load_weights_if_possible(model, init_weight_path=None):
    """Loads model weights when it is provided."""
    initial_epoch = 0
    if init_weight_path:
        tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
        model.load_weights(init_weight_path).expect_partial()
        initial_epoch = int(init_weight_path.split('/')[-1].split('-')[-1].split('.')[0])
    else:
        print("Weights not loaded from path:{}".format(init_weight_path))

    return initial_epoch

def create_optimizer(params):
    """Creates optimizer."""
    opt = tf.keras.optimizers.Adam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    return opt


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
    print(propnet_weight_path)
    propnet_weights = cPickle.load(open(propnet_weight_path, 'rb'))

    model_layers = [l for l in model.layers]
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
    print(simnet_weight_path)
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


def get_train_model(params):
    model_name = "optgen_%s" % params["model_version"]
    model = eval(model_name).create_model(params, is_train=True)
    print("=========================== %s created!! ==========================" % model_name)

    opt = create_optimizer(params)
    model.compile(opt)
    initial_epoch = load_weights_if_possible(model, tf.train.latest_checkpoint(params["model_dir"]))
    initial_step = initial_epoch * params["steps_between_evals"]
    model.summary()

    if params["use_propnet"] == 1:
        print("========================== Transferring PropNET weights....==========================")
        transfer_weights_from_propnet(model, params)
    if params["use_simnet"] == 1:
        print("========================== Transferring SimNET weights....==========================")
        transfer_weights_from_simnet(model, params)


    # else:
    #     print("========================== NOT Transferring MT weights, just randomly initialized....===============")
    return model, initial_epoch, initial_step


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
        if params["use_propnet"] == 1:
            print("========================== Transferring PropNET weights....==========================")
            transfer_weights_from_propnet(get_dev_model.model, params)
        if params["use_simnet"] == 1:
            print("========================== Transferring SimNET weights....==========================")
            transfer_weights_from_simnet(get_dev_model.model, params)


    else:
        print("dev_model is reused!!!")

    return get_dev_model.model


def get_vjtnn_data(prop, split):
    # split = valid, test, train_pairs
    datapath = '%s/vjtnn/%s/%s.txt' % (args.p, prop, split)

    if split=="train_pairs":
        data = []
        with open(datapath) as f:
            for line in f:
                data.append(line.split()[0].strip())
                data.append(line.split()[1].strip())
    else:
        with open(datapath) as f:
            data = [line.split()[0].strip() for line in f]

    return data


@static_var("batches", None)
def get_dev_batches(moltokenizer):
    if get_dev_batches.batches is None:
        # print("[property_dic] loading %s.%s..." % (prop, split))
        # test_smiles_list = get_vjtnn_data(prop, split)

        file_name = '%s/v10.all_kinds_of_smiles.cpkl' % args.p
        (names, smiles_dataset) = cPickle.load(open(file_name, 'rb'))
        trn_smiles, dev_logp_smiles, dev_qed_smiles, dev_drd2_smiles, dev_all_smiles, tst_logp_smiles, tst_qed_smiles, \
        tst_drd2_smiles, tst_all_smiles, all_smiles = smiles_dataset

        file_name = '%s/v10.property_for_all_smiles.cpkl' % args.p
        property_dic = cPickle.load(open(file_name, 'rb'))

        smiles_list = []
        ids_list = []
        logp_list = []
        qed_list = []
        drd2_list = []

        desired_logp_list = []
        desired_qed_list = []
        desired_drd2_list = []

        for smiles in dev_all_smiles:
            smiles_list.append(smiles)
            ids = moltokenizer.encode(smiles)
            # ids_list.append(test_smiles[smiles]['ids'])
            ids_list.append(ids)
            logp_list.append(property_dic[smiles]['logP'])
            qed_list.append(property_dic[smiles]['qed'])
            drd2_list.append(property_dic[smiles]['drd2'])

            desired_logp_list.append(property_dic[smiles]['logP'] + 3.5)  # 2.33 ± 1.24, 3.55 ± 1.67
            desired_qed_list.append(0.91)  # QED ∈ [0.9, 1.0]
            desired_drd2_list.append(0.51)  # DRD2 > 0.5

        batches = []
        property_x = np.stack([np.array(logp_list), np.array(qed_list), np.array(drd2_list)], axis=1)
        property_desired = np.stack(
            [np.array(desired_logp_list), np.array(desired_qed_list), np.array(desired_drd2_list)],
            axis=1)

        n = len(dev_all_smiles)
        dev_iter = 6
        dev_batch = int(n/dev_iter)
        for idx in range(dev_iter):  # 1038/6 = 173
            start_i = idx * dev_batch
            end_i = (idx + 1) * dev_batch
            smiles_batch = smiles_list[start_i:end_i]
            ids_batch = np.array(tf.keras.preprocessing.sequence.pad_sequences(
                ids_list[start_i:end_i], dtype="int64", padding="post"))

            property_x_batch = property_x[start_i:end_i]
            property_desired_batch = property_desired[start_i:end_i]

            one_batch = [smiles_batch, ids_batch, property_x_batch, property_desired_batch]
            batches.append(one_batch)

        get_dev_batches.batches = batches
        print("dev_batches is created!!!")

    else:
        print("dev_batches is reused!!!")

    return get_dev_batches.batches


def get_train_ds():
    train_ds = data_pipeline.train_input_fn(params)
    train_ds = train_ds.map(data_pipeline.map_data_for_optgen_fn,
                            num_parallel_calls=params["num_parallel_calls"])

    return train_ds


def create_callbacks(cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    # callbacks = misc.get_callbacks()
    callbacks = []
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))

    return callbacks


def get_scores(src, trg):
    sim = similarity(src, trg)
    # if sim < 0.4 or sim >= 1.0:
    if sim >= 1.0:
        sim = 0

    logp_improvement = penalized_logp(trg) - penalized_logp(src)
    qed_score = qed(trg)
    drd2_score = drd2(trg)

    return sim, logp_improvement, qed_score, drd2_score


def evaluate_dev(params, trn_model):
    """Evaluates the model."""

    predict_model = get_dev_model(params)
    model_name = "optgen_%s" % params["model_version"]

    trn_optgen_layer_index = get_layer_index(trn_model, name=model_name)
    dev_optgen_layer_index = get_layer_index(predict_model, name=model_name)
    predict_model.layers[dev_optgen_layer_index].set_weights(trn_model.layers[trn_optgen_layer_index].get_weights())
    print("trn_optgen_layer_index(%d, %s), dev_optgen_layer_index(%d, %s)" %
          (trn_optgen_layer_index, trn_model.layers[trn_optgen_layer_index].name,
           dev_optgen_layer_index, predict_model.layers[dev_optgen_layer_index].name))
    print("set weights for (%s) layer" % model.layers[dev_optgen_layer_index].name)

    # initial_epoch = load_weights_if_possible(predict_model, tf.train.latest_checkpoint(params["model_dir"]))
    predict_model.summary()

    moltokenizer = Moltokenizer(params["vocab_file"], model_version=params["model_version"])
    batches = get_dev_batches(moltokenizer)


    src_list = []
    trg_list = []
    sim_list = []
    logpi_list = []
    qed_list = []
    drd2_list = []

    if params["model_version"] == "v3":
        mse_px_list = []
        mse_py_list = []
        mse_npx_list = []

    for batch in tqdm.tqdm(batches):
        smiles_batch, ids_batch, property_x_batch, property_desired_batch = batch


        val_outputs, scores = \
            predict_model.predict([ids_batch, property_x_batch, property_desired_batch])

        length = len(val_outputs)
        for idx, j in enumerate(range(length)):
            src = smiles_batch[idx]
            src_list.append(src)
            trg = moltokenizer.decode(val_outputs[j])
            trg_list.append(trg)

            if len(trg) == 0:
                # print("no translation!!")
                sim_list.append(0.0)
                logpi_list.append(0.0)
                qed_list.append(0.0)
                drd2_list.append(0.0)

            else:
                sim, logp_improvement, qed_score, drd2_score = get_scores(src, trg)
                sim_list.append(sim)
                logpi_list.append(logp_improvement)
                qed_list.append(1.0 if qed_score>=0.9 else 0.0)
                drd2_list.append(1.0 if drd2_score>0.5 else 0.0)


    sims = np.array(sim_list)
    print("invalid(%d):valid(%d) similar(%d)" % (sum(sims == 0.0), sum(sims > 0.0), sum(sims >= 0.4)))
    print("valid mean(%f)" % np.mean(sims[np.where(sims > 0)]))


    return np.mean(sim_list), np.std(sim_list), np.mean(logpi_list), np.std(logpi_list), np.mean(qed_list), \
           np.mean(drd2_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='../data', help='Base Directory.')
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str)
    parser.add_argument('-lr', default="2", type=str) # 1e-3
    parser.add_argument('-e', type=int, default=500, help='epoch num')
    parser.add_argument('-l', default=150, type=int, help='max length')
    parser.add_argument('-he', default=8, type=int, help='num_heads')
    parser.add_argument('-hi', default=4, type=int, help='num_hidden_layers')
    parser.add_argument('-f', default=256, type=int, help='filter_size')
    parser.add_argument('-b', default=4096, type=int, help='batch_size') # 4096, 2048
    # v1 = original, v2 = MT conpatible, v3 = disentangled property encoding
    parser.add_argument('-v', default=21, type=int, help='model version')
    parser.add_argument('-pnet', default=1, type=int, help='if propnet used')
    parser.add_argument('-snet', default=0, type=int, help='if simnet used')
    parser.add_argument('-bf', default=0, type=int, help='if use_beam_filter')
    parser.add_argument('-t', default=0, type=int, help='attempt')



    args, unparsed = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    model_version = "v%d" % args.v
    base_path = args.p

    n_samples = 10827615
    
    data_dir = "%s/v10_trn" % base_path

    iterations = args.e
    steps_between_evals = n_samples//args.b
    train_steps = steps_between_evals * iterations
    validation_steps = 32
    batch_size = args.b

    model_dir = "%s/model.vv%d.vnet%d.pnet%d.snet%d.head%d.hid%d.fil%d.batch%d.lr%s.t%d" % (base_path, args.v,
                                                                                            args.vnet, args.pnet,
                                                                                            args.snet, args.he, args.hi,
                                                                                            args.f, args.b, args.lr,
                                                                                            args.t)

    propnet_weight_path = "%s/v30.propnet.weights.cpkl" % base_path
    simnet_weight_path = "%s/v30.simnet.weights.cpkl" % base_path

    optgen_config_file = "%s/../config/optgen_config.json" % base_path
    params = optgen_v2.load_config(optgen_config_file)
    params["steps_between_evals"] = steps_between_evals
    params["batch_size"] = batch_size
    params["train_steps"] = train_steps
    params["validation_steps"] = validation_steps
    params["model_dir"] = model_dir
    params["propnet_weight_path"] = propnet_weight_path
    params["simnet_weight_path"] = simnet_weight_path

    params["num_heads"] = args.he
    params["num_hidden_layers"] = args.hi
    params["filter_size"] = args.f
    params["learning_rate"] = float(args.lr)
    params["model_version"] = model_version

    params["n_samples"] = n_samples
    params["iterations"] = iterations
    params["data_dir"] = data_dir
    params["max_length"] = args.l

    params["use_propnet"] = args.pnet
    params["use_simnet"] = args.snet
    params["use_beam_filter"] = args.bf

    params["vocab_file"] = "%s/optgen_vocab.txt" % base_path
    params["vocab_size"] = 71


    print("========================================[params]========================================")
    for k,v in params.items():
        print(k, ":" ,v)
    print("========================================[params]========================================")



    model, initial_epoch, initial_step = get_train_model(params)
    train_ds = get_train_ds()

    callbacks = create_callbacks(params["model_dir"], initial_step, params)

    if train_steps < steps_between_evals:
        steps_between_evals = train_steps
    iterations = (train_steps - initial_step) // steps_between_evals
    sys.stdout.flush()

    mse_px_list = []
    mse_npx_list = []
    mse_py_list = []
    for i in range(1, iterations + 1):
        print("Start train iteration:{}/{}".format(i + initial_epoch, initial_epoch + iterations))
        history = model.fit(
            train_ds,
            initial_epoch=i - 1 + initial_epoch,
            epochs=i + initial_epoch,
            steps_per_epoch=steps_between_evals,
            callbacks=callbacks,
            verbose=1)
        print("End train iteration:{}/{} global step:{}".format(
            i + initial_epoch,
            iterations + initial_epoch,
            (i + initial_epoch) * steps_between_evals))

        sys.stdout.flush()
    print(1)

