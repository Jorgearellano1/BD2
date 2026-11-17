from enum import Enum, auto


class TokenType(Enum):
    # Reserved words
    CREATE = auto()
    TABLE = auto()
    INDEX = auto()
    ON = auto()
    INSERT = auto()
    INTO = auto()
    VALUES = auto()

    DELETE = auto()

    ALTER = auto()
    ADD = auto()
    UNIQUE = auto()

    SELECT = auto()
    FROM = auto()
    WHERE = auto()

    FORCE = auto()
    SEQSCAN = auto()
    USING = auto()

    HEAP = auto()
    SEQUENTIAL = auto()

    CSV = auto()
    DELIMITER = auto()

    PRIMARY = auto()
    KEY = auto()

    BTREE = auto()
    HASH = auto()
    RTREE = auto()
    WITHIN_RADIUS = auto()
    KNN = auto()

    # Modulo textual
    INVERTED = auto()
    WITH = auto()
    MATCH = auto()
    ORDER = auto()
    BY = auto()
    LIMIT = auto()
    ASC = auto()
    DESC = auto()
    AS = auto()
    COSINE_SCORE = auto()

    # Modulo vectorial
    VECTOR = auto()
    IVF = auto()
    SET = auto()
    DOT = auto()
    L2_DIST = auto()
    COSINE_DIST = auto()
    IP_DIST = auto()

    AND = auto()
    OR = auto()
    NOT = auto()

    # SQL types
    INT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    CHAR = auto()

    # Identifiers and literals
    ID = auto()
    INT_LITERAL = auto()
    REAL_LITERAL = auto()
    STRING_LITERAL = auto()

    # Symbols
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()
    STAR = auto()

    # Operators
    EQ = auto()
    NEQ = auto()
    NEQ2 = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Special tokens
    EOF = auto()
    ERR = auto()


class Token:
    def __init__(self, token_type: TokenType, text: str = ""):
        self.type = token_type
        self.text = text

    def __str__(self) -> str:
        if self.type == TokenType.EOF:
            return "TOKEN(EOF)"

        return f'TOKEN({self.type.name}, "{self.text}")'

    def __repr__(self) -> str:
        return str(self)