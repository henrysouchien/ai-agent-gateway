from agent_gateway.agent_session_log_cache import ActiveFileOffsetCache


def test_active_file_offset_cache_tracks_bounds_and_completion() -> None:
  cache = ActiveFileOffsetCache()
  identity = (1, 2, 300, 4)

  assert cache.starting_offset_for_seq(None, active_identity=identity) == 0
  assert cache.starting_offset_for_seq(1, active_identity=identity) == 0

  cache.update(seq=10, offset=120, active_identity=identity)
  cache.update(seq=10, offset=140, active_identity=identity)
  cache.update(seq=11, offset=180, active_identity=identity)

  assert cache.starting_offset_for_seq(10, active_identity=identity) == 120
  assert cache.starting_offset_for_seq(11, active_identity=identity) == 180
  assert cache.starting_offset_for_seq(12, active_identity=identity) == 180

  cache.mark_complete(active_identity=identity, current_identity=identity, file_size=300)

  assert cache.starting_offset_for_seq(12, active_identity=identity) == 300


def test_active_file_offset_cache_invalidates_when_identity_changes() -> None:
  cache = ActiveFileOffsetCache()
  first_identity = (1, 2, 300, 4)
  second_identity = (1, 3, 25, 5)

  cache.update(seq=10, offset=120, active_identity=first_identity)

  assert cache.starting_offset_for_seq(10, active_identity=first_identity) == 120
  assert cache.starting_offset_for_seq(10, active_identity=second_identity) == 0
