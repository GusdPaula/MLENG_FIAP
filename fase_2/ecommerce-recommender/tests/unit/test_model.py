import torch
from recommender.models.ncf import NCFModel


def test_ncf_forward_shape():
    model = NCFModel(
        num_users=100, num_items=50, embedding_dim=16, hidden_layers=[32, 16]
    )
    users = torch.randint(0, 100, (8,))
    items = torch.randint(0, 50, (8,))

    output = model(users, items)

    assert output.shape == (8,)


def test_ncf_output_range():
    model = NCFModel(
        num_users=100, num_items=50, embedding_dim=16, hidden_layers=[32, 16]
    )
    users = torch.randint(0, 100, (32,))
    items = torch.randint(0, 50, (32,))

    output = model(users, items)

    assert (output >= 0).all()
    assert (output <= 1).all()


def test_ncf_single_sample():
    model = NCFModel(num_users=10, num_items=10, embedding_dim=8, hidden_layers=[16])
    user = torch.tensor([0])
    item = torch.tensor([0])

    output = model(user, item)

    assert output.shape == (1,)
    assert output.item() >= 0 and output.item() <= 1
