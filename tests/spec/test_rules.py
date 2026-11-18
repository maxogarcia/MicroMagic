"""Rules tests for the core turn loop: lands, mana, creatures, combat, and game end."""

from micromagic.env import spec
from micromagic.env.spec import ATTACK, BLOCK, CARD_IDS, EMPTY, PASS, PLAY_HAND, Phase
from micromagic.env.state import GameState


LAND = CARD_IDS["Mountain"]
ONE_ONE = CARD_IDS["Suntail Hawk"]
TWO_TWO = CARD_IDS["Grizzly Bears"]
THREE_THREE = CARD_IDS["Trained Armodon"]
TWO_FOUR = CARD_IDS["Giant Spider"]


def make_state(hand0: list[int], hand1: list[int] | None = None) -> GameState:
    """A fresh game at player 0's first main phase with the given hands."""
    state = GameState(seed=0)
    for player, cards in enumerate((hand0, hand1 or [])):
        state.hand[player] = EMPTY
        state.hand[player, : len(cards)] = cards
        state.hand_count[player] = len(cards)
    return state


def put_creature(state: GameState, player: int, slot: int, card_id: int, sick: bool = False) -> None:
    state.creature[player, slot] = card_id
    state.sick[player, slot] = sick


def test_opening_hands_are_seven_and_first_player_skips_draw() -> None:
    state = GameState(seed=0)
    assert state.hand_count.tolist() == [7, 7]
    assert state.active == 0 and state.phase == Phase.MAIN1 and state.turn == 1


def test_one_land_per_turn() -> None:
    state = make_state([LAND, LAND])
    assert state.legal_actions()[PLAY_HAND.start]
    state.step(PLAY_HAND.start)
    assert state.lands_total[0] == 1 and state.hand_count[0] == 1
    assert not state.legal_actions()[PLAY_HAND.start]


def test_creature_needs_enough_untapped_lands() -> None:
    state = make_state([TWO_TWO])
    state.lands_total[0] = 1
    assert not state.legal_actions()[PLAY_HAND.start]
    state.lands_total[0] = 2
    state.step(PLAY_HAND.start)
    assert state.creature[0, 0] == TWO_TWO and state.lands_tapped[0] == 2


def test_summoning_sick_creature_cannot_attack() -> None:
    state = make_state([ONE_ONE])
    state.lands_total[0] = 1
    state.step(PLAY_HAND.start)
    state.step(PASS)  # leaving main 1 skips combat: nothing can attack
    assert state.phase == Phase.MAIN2


def test_draw_skipped_when_hand_is_full() -> None:
    state = make_state([], [LAND] * 7)
    library_before = state._library_count(1)
    state.step(PASS)  # main 1 -> main 2 (no attackers)
    state.step(PASS)  # end turn -> player 1 draws
    assert state.active == 1 and state.hand_count[1] == 7
    assert state._library_count(1) == library_before


def test_drawing_from_empty_library_loses() -> None:
    state = make_state([], [])
    state.library_top[1] = len(state.library[1])
    state.step(PASS)
    state.step(PASS)
    assert state.is_terminal() and state.winner() == 0


def test_unblocked_attacker_deals_damage() -> None:
    state = make_state([])
    put_creature(state, 0, 0, THREE_THREE)
    state.step(PASS)
    assert state.phase == Phase.DECLARE_ATTACKERS
    state.step(ATTACK.start)
    state.step(PASS)  # defender has no creatures, so combat resolves
    assert state.life[1] == spec.STARTING_LIFE - 3
    assert state.tapped[0, 0] and state.phase == Phase.MAIN2


def test_defender_chooses_blocks_and_trades() -> None:
    state = make_state([])
    put_creature(state, 0, 0, TWO_TWO)
    put_creature(state, 1, 0, TWO_TWO)
    state.step(PASS)
    state.step(ATTACK.start)
    state.step(PASS)
    assert state.phase == Phase.DECLARE_BLOCKERS and state.current_player() == 1
    assert state.legal_actions()[BLOCK.start]
    state.step(BLOCK.start)
    assert state.current_player() == 0 and state.phase == Phase.MAIN2
    assert state.creature[0, 0] == EMPTY and state.creature[1, 0] == EMPTY
    assert state.life[1] == spec.STARTING_LIFE


def test_bigger_blocker_survives_and_damage_wears_off() -> None:
    state = make_state([])
    put_creature(state, 0, 0, TWO_TWO)
    put_creature(state, 1, 0, TWO_FOUR)
    state.step(PASS)
    state.step(ATTACK.start)
    state.step(PASS)
    state.step(BLOCK.start)
    assert state.creature[0, 0] == EMPTY
    assert state.creature[1, 0] == TWO_FOUR and state.damage[1, 0] == 2
    state.step(PASS)  # end turn
    assert state.damage[1, 0] == 0


def test_lethal_damage_ends_game() -> None:
    state = make_state([])
    put_creature(state, 0, 0, THREE_THREE)
    state.life[1] = 3
    state.step(PASS)
    state.step(ATTACK.start)
    state.step(PASS)
    assert state.is_terminal() and state.winner() == 0


def test_turn_cap_ends_in_draw() -> None:
    state = make_state([])
    state.turn = spec.MAX_TURNS
    state.step(PASS)
    state.step(PASS)
    assert state.is_terminal() and state.is_truncated() and state.winner() is None
