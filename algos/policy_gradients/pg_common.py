import torch.nn.functional as F


def calculate_entropy(logits_v, log_prob_v, beta=0.01):
    """
	calculate the entropy for the policy gradient method determining how certain an agent is about the 
	action they are taking

	Args:
		logits_v: output of the network
		log_prob_v: probability distribution of taking each of the actions from the network

	Returns:
		entropy_loss_v:  entropy loss
		prob_v: probability distribution of the network outputs
	"""
    prob_v = F.softmax(logits_v, dim=1)
    entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
    entropy_loss_v = -beta * entropy_v
    return entropy_loss_v, prob_v


def calculate_kl_divergence(new_logits_v, prob_v):
    """
	calculates the KL divergence term of the new and old probability distributions

	Args:
		new_logits_v: ouputs from the neural network
		prob_v: old probability distribution from neural network

	Return:
		kl_div_v: KL divergence term
	"""
    new_prob_v = F.softmax(new_logits_v, dim=1)
    kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
    return kl_div_v
