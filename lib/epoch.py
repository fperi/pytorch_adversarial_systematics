import torch
import random


def adjust_learning_rate(optimizer, epoch):

    '''
    Function scales down the learning rate with time

    :param optimizer: network optimiser
    :param epoch: epoch

    :return: optmiser with reduced learning rate
    '''

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.99 ** (epoch // 40))

    return optimizer


def train_epoch(which, df, features, dataset,
                netD, optimiserD, criterionD,
                netA, optimiserA, criterionA,
                penalty=0, single_batch=False,
                batch_size=256):

    '''
    Function allows to train the input neural network,
    either singularly or together in adversarial mode.

    :param which: which network to train
        'D' discriminant only
        'A' adversarial term only
        'both' full adversarial mode
    :param netD: discriminant nnet
    :param optimiserD: optimiser for netD
    :param criterionD: loss criterion for netD
    :param netA: adversarial term nnet
    :param optimiserA: optimiser for neta
    :param criterionA: loss criterion for netA
    :param dataset: input dataset
    :param penalty: penalty for the adversarial term
    :param single_batch: whether to train over a single batch instead
        than a full epoch
    :param batch_size: batch size
        (a big batch size to let the adv network pick up
        the large scale differences)

    :return: the modified dataframe,
        the two neural networks, D and A,
        and the metrics output of the training,
        loss of D, loss of A and combined loss
    '''
    # set seeds
    RANDOM_STATE = 42
    torch.manual_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # process input dataset
    X_train_t, Y_train_t, X_test_t, Y_test_t, AY_train_t, AY_test_t = dataset
    AY_test_t = AY_test_t.long().squeeze(1)

    # initialise output metrics
    dloss = []
    aloss = []
    closs = []

    dloss_test = []
    aloss_test = []
    closs_test = []

    # check whether to loop on a full epoch or a single batch
    if single_batch:
        limitl = random.randint(0, len(X_train_t) - batch_size)
        limith = limitl + batch_size
    else:
        limitl = 0
        limith = len(X_train_t)

    # loop over batches
    for b_i in range(limitl, limith, batch_size):

        # select batch data
        X_batch = X_train_t[b_i:b_i + batch_size, :]
        Y_batch = Y_train_t[b_i:b_i + batch_size, :]
        AY_batch = AY_train_t[b_i:b_i + batch_size, :]
        AY_batch = AY_batch.long().squeeze(1)

        # discriminant training
        if which == 'D':

            optimiserD.zero_grad()
            netD.train()
            hatD = netD(X_batch)
            lossD = criterionD(hatD, Y_batch)
            lossD.backward()
            optimiserD.step()
            netD.eval()

            if b_i % 10 == 0 or b_i == limitl:
                dloss.append(lossD.item())
                dloss_test.append(criterionD(netD(X_test_t), Y_test_t))

        # adversarial term only training
        if which == 'A':

            optimiserA.zero_grad()
            netA.train()
            hatD = netD(X_batch)
            hatA = netA(hatD[AY_batch > 0])
            lossA = criterionA(torch.log(hatA), AY_batch[AY_batch > 0])
            lossA.backward()
            optimiserA.step()
            netA.eval()

            if b_i % 10 == 0 or b_i == limitl:
                aloss.append(lossA.item())
                aloss_test.append(
                    criterionA(torch.log(netA(netD(X_test_t)[AY_test_t > 0])),
                               AY_test_t[AY_test_t > 0]))

        # full adversarial training
        if which == 'both':

            # discriminant
            optimiserD.zero_grad()
            netD.train()
            hatD = netD(X_batch)
            lossD = criterionD(hatD, Y_batch)
            lossD.backward(retain_graph=True)

            if b_i % 10 == 0 or b_i == limitl:
                dloss.append(lossD.item())
                dloss_test.append(criterionD(netD(X_test_t), Y_test_t))

            # adv term
            optimiserA.zero_grad()
            hatA = netA(hatD[AY_batch > 0])
            lossA = - penalty * criterionA(torch.log(hatA),
                                           AY_batch[AY_batch > 0])
            lossA.backward()
            optimiserD.step()
            netD.eval()

            if b_i % 10 == 0 or b_i == limitl:
                aloss.append(lossA.item() / (-penalty))
                closs.append(dloss[-1] - penalty * aloss[-1])
                aloss_test.append(
                    criterionA(torch.log(netA(netD(X_test_t)[AY_test_t > 0])),
                               AY_test_t[AY_test_t > 0]))
                closs_test.append(dloss_test[-1] - penalty * aloss_test[-1])

    # add prediction column to df
    if which == 'D' or which == 'both':
        X_temp = (df[features]).values
        X_temp_t = torch.FloatTensor(X_temp)
        df['predictionD'] = netD(X_temp_t).detach().numpy()
        if which == 'both':
            X_temp = (df['predictionD']).values.reshape(-1, 1)
            X_temp_t = torch.Tensor(X_temp)
            df['predictionA'] = netA(X_temp_t).detach().numpy().tolist()

    return dloss_test, aloss_test, closs_test
