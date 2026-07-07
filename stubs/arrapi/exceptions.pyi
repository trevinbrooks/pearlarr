"""The arrapi exception surface the CLI catches (scope notes in __init__.pyi).

``ConnectionFailure`` (unreachable url) and ``Unauthorized`` (rejected API key)
are raised from the ``SonarrAPI``/``RadarrAPI`` constructors' status ping; the
CLI maps both to clean one-line config errors.
"""

class ArrException(Exception): ...
class ConnectionFailure(ArrException): ...
class Unauthorized(ArrException): ...
