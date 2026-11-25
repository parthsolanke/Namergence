import torch

from model import RNN


def test_init_hidden_returns_zero_tensor():
    model = RNN(input_size=5, hidden_size=7, output_size=3)

    hidden = model.init_hidden()

    assert hidden.shape == (1, 7)
    assert torch.count_nonzero(hidden).item() == 0


def test_forward_returns_log_probabilities_per_category():
    model = RNN(input_size=5, hidden_size=7, output_size=3)
    input_tensor = torch.zeros(1, 5)

    output, next_hidden = model(input_tensor, model.init_hidden())

    assert output.shape == (1, 3)
    assert next_hidden.shape == (1, 7)
    assert torch.all(output <= 0)
    assert torch.allclose(
        torch.exp(output).sum(dim=1), torch.ones(1), atol=1e-6
    )


def test_forward_can_step_through_a_sequence():
    model = RNN(input_size=3, hidden_size=4, output_size=2)
    hidden = model.init_hidden()
    sequence = torch.zeros(5, 1, 3)

    for step in sequence:
        output, hidden = model(step, hidden)

    assert output.shape == (1, 2)
    assert hidden.shape == (1, 4)


def test_forward_does_not_mutate_previous_hidden_state():
    model = RNN(input_size=4, hidden_size=6, output_size=2)
    hidden = model.init_hidden()
    snapshot = hidden.clone()

    model(torch.ones(1, 4), hidden)

    assert torch.equal(hidden, snapshot)
