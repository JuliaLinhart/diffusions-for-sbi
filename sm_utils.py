import random
import torch

from tqdm import tqdm


def train(
    model, dataset, loss_fn, n_epochs=5000, lr=3e-4, batch_size=32, classifier_free_guidance=0., track_loss=False, validation_split=0,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                    multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    # get train and validation loaders
    train_loader, val_loader = get_dataloaders(
        dataset, batch_size=batch_size, validation_split=validation_split
    )

    train_losses = []
    val_losses = []
    with tqdm(range(n_epochs), desc="Training epochs") as tepoch:
        for _ in tepoch:
            train_loss = 0
            for data in train_loader:
                # get batch data
                if len(data) > 1:
                    theta, x, kwargs_sn = get_batch_data(data, classifier_free_guidance)
                    kwargs_sn["x"] = x
                else:
                    theta = data[0]
                    kwargs_sn = {}
                # train step
                opt.zero_grad()
                loss = loss_fn(theta, **kwargs_sn)
                loss.backward()
                opt.step()
                ema_model.update_parameters(model)

                # update loss
                train_loss += loss.detach().item() * theta.shape[0]  # unnormalized loss
            train_loss /= len(dataset) * (1 - validation_split)  # normalized loss
            train_losses.append(train_loss)

            # validation loop
            if val_loader is not None:
                with torch.no_grad():
                    val_loss = 0
                    for data in val_loader:
                        # get batch data
                        if len(data) > 1:
                            theta, x, kwargs_sn = get_batch_data(data, classifier_free_guidance)
                            kwargs_sn["x"] = x
                        else:
                            theta = data[0]
                            kwargs_sn = {}
                        # validation step
                        loss = loss_fn(theta, **kwargs_sn)
                        
                        # update loss
                        val_loss += (
                            loss.detach().item() * theta.shape[0]
                        )  # unnormalized loss
                    val_loss /= len(dataset) * validation_split  # normalized loss
                    val_losses.append(val_loss)

                tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=lr)
            else:
                tepoch.set_postfix(train_loss=train_loss, lr=lr)

    if track_loss:
        return ema_model, train_losses, val_losses
    else:
        return ema_model

# Training with validation and early stopping as in
# https://github.com/smsharma/mining-for-substructure-lens/blob/master/inference/trainer.py


def train_with_validation(
    model,
    dataset,
    loss_fn,
    n_epochs=100,
    lr=1e-4,
    batch_size=128,
    lr_decay=1,
    lr_update_freq=2000,
    validation_split=0.2,
    early_stopping=False,
    patience=20000,
    min_nb_epochs=100,
    classifier_free_guidance=False,
):
    # get train and validation loaders
    train_loader, val_loader = get_dataloaders(
        dataset, batch_size=batch_size, validation_split=validation_split
    )

    # set up optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                    multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    

    # start training
    train_losses, val_losses = [], []
    best_loss, best_model, best_epoch = None, None, None

    with tqdm(range(n_epochs), desc="Training epochs") as tepoch:
        for e in tepoch:
            # update learning rate
            # lr_e = update_lr(lr, lr_decay, e, n_epochs)
            if (e + 1) % lr_update_freq == 0 and (lr_decay < 1):
                lr = lr * lr_decay
                set_lr(opt, lr)

            # training loop
            train_loss = 0
            for data in train_loader:
                # get batch data
                if len(data) > 1:
                    theta, x, kwargs_sn = get_batch_data(data, classifier_free_guidance)
                    kwargs_sn["x"] = x
                else:
                    theta = data[0]
                    kwargs_sn = {}
                # training step
                opt.zero_grad()
                loss = loss_fn(theta, **kwargs_sn)
                loss.backward()
                opt.step()
                ema_model.update_parameters(model)

                # update loss
                train_loss += loss.detach().item() * theta.shape[0]  # unnormalized loss
            train_loss /= len(dataset) * (1 - validation_split)  # normalized loss
            train_losses.append(train_loss)

            # validation loop
            if val_loader is not None:
                with torch.no_grad():
                    val_loss = 0
                    for data in val_loader:
                        # get batch data
                        if len(data) > 1:
                            theta, x, kwargs_sn = get_batch_data(data, classifier_free_guidance)
                            kwargs_sn["x"] = x
                        else:
                            theta = data[0]
                            kwargs_sn = {}
                        # validation step
                        loss = loss_fn(theta, **kwargs_sn)

                        # update loss
                        val_loss += (
                            loss.detach().item() * theta.shape[0]
                        )  # unnormalized loss
                    val_loss /= len(dataset) * validation_split  # normalized loss
                    val_losses.append(val_loss)
                tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=lr)
            else:
                tepoch.set_postfix(loss=train_loss, lr=lr)

            # early stopping
            if early_stopping:
                assert (
                    validation_split > 0
                ), "validation data is required for early stopping"
                if best_loss is None or val_loss < best_loss and e > min_nb_epochs:
                    best_loss = val_loss
                    best_model = ema_model
                    best_epoch = e
                elif e - best_epoch > patience:
                    break

        if early_stopping:
            ema_model = best_model
            e = best_epoch
            print(
                f"EARLY STOPPING: best validation loss {best_loss} at epoch {best_epoch}"
            )
            print("Did not improve for {} epochs".format(min(patience, n_epochs - e)))

    return ema_model, train_losses, val_losses, e


def get_dataloaders(dataset, batch_size, validation_split):
    if validation_split > 0:
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset, [1 - validation_split, validation_split]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, shuffle=True
        )
    else:
        dataset_train = dataset
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        val_loader = None
    return train_loader, val_loader


def update_lr(lr, lr_decay, epoch, n_epochs):
    return lr * lr_decay ** (epoch / (n_epochs - 1))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_batch_data(data, clf_free_guidance):
    # default: npse without n
    theta, x = data[0], data[1]
    n = None
    kwargs_sn = {}

    if len(data) > 2:
        n = data[2]
        kwargs_sn = {"n": n}

    # to learn the diffused prior score via classifier free guidance: 
    # set context to zero 20% of the time
    if random.random() < clf_free_guidance:
        x = torch.zeros_like(x)  # zero context
        kwargs_sn = {}
        if n is not None:
            n = torch.zeros((x.shape[0], 1))  # zero set size
            kwargs_sn["n"] = n

    return theta, x, kwargs_sn