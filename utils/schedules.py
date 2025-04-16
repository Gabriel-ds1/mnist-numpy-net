def exponential_decay_schedule(epoch, initial_lr=0.01, final_lr=0.001, total_epochs=100):
    """
    Returns the learning rate for a given epoch based on an exponential decay schedule.
    
    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
    
    Returns:
        float: The learning rate for the current epoch.
    """
    # Calculate the decay rate so that at epoch total_epochs, lr == final_lr
    decay_rate = (final_lr / initial_lr) ** (1 / total_epochs)
    # Calculate the learning rate for the current epoch
    lr = initial_lr * (decay_rate ** epoch)
    return lr

def step_decay_schedule(epoch, initial_lr=0.01, final_lr=0.001, total_epochs=100, step_size=30):
    """
    Returns the learning rate for a given epoch based on a step decay schedule.

    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
        step_size (int): Number of epochs between each decay.

    Returns:
        float: The learning rate for the current epoch.
    """
    # Estimate how many times to decay over the total_epochs
    num_decays = total_epochs // step_size
    # Compute the decay factor needed
    decay_factor = (final_lr / initial_lr) ** (1 / max(1, num_decays))
    # Compute how many steps have passed
    num_steps = epoch // step_size
    lr = initial_lr * (decay_factor ** num_steps)
    return lr