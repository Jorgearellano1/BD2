from ast import DeleteStm
from ast import SelectStm
from ast import AlterStm
from ast import InsertStm
from ast import CsvSource
from scanner import Scanner
from token import Token, TokenType
from ast import Program, Stm, CreateIndexStm, IndexType, CreateTableStm, StorageType, ColumnDef, DataType, Exp, LiteralExp
from ast import BinaryOp, UnaryOp
from ast import BinaryExp, UnaryExp, LiteralExp, IdExp

class Parser:
    def __init__(self, scanner: Scanner):
        self.scanner = scanner
        self.previous: Token | None = None
        self.current: Token = scanner.nextToken()

        if self.current.type == TokenType.ERR:
            raise SyntaxError(
                f"Error lexico: token invalido '{self.current.text}'"
            )

    # Metodos auxiliares
    def check(self, token_type: TokenType) -> bool:
        return self.current.type == token_type

    def match(self, token_type: TokenType) -> Token:
        if self.check(token_type):
            return self.advance()

        raise SyntaxError(
            f"Error sintactico: se esperaba {token_type.name}, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    def advance(self) -> Token:
        if self.is_at_end():
            return self.current

        self.previous = self.current
        self.current = self.scanner.nextToken()

        if self.current.type == TokenType.ERR:
            raise SyntaxError(
                f"Error lexico: token invalido '{self.current.text}'"
            )

        return self.previous

    def is_at_end(self) -> bool:
        return self.current.type == TokenType.EOF

    # Reglas gramaticales
    def parse_program(self) -> Program:
        statements: list[Stm] = []

        while not self.is_at_end():
            statements.append(self.parse_statement())

        return Program(statements)

    def parse_statement(self) -> Stm:
        if self.check(TokenType.CREATE):
            self.match(TokenType.CREATE)
            return self.parse_create_stmt_rest()

        elif self.check(TokenType.INSERT):
            self.match(TokenType.INSERT)
            return self.parse_insert_stmt_rest()

        elif self.check(TokenType.ALTER):
            self.match(TokenType.ALTER)
            return self.parse_alter_stmt_rest()

        elif self.check(TokenType.SELECT):
            self.match(TokenType.SELECT)
            return self.parse_select_stmt_rest()

        elif self.check(TokenType.DELETE):
            self.match(TokenType.DELETE)
            return self.parse_delete_stmt_rest()

        raise SyntaxError(
            f"Error sintactico: token inesperado "
            f"{self.current.type.name} ('{self.current.text}')"
        )

    # CREATE

    def parse_create_stmt_rest(self) -> Stm:
        if self.check(TokenType.TABLE):
            self.match(TokenType.TABLE)

            name = self.match(TokenType.ID).text

            return self.parse_create_table_body(name)

        elif self.check(TokenType.INDEX):
            self.match(TokenType.INDEX)

            name = self.match(TokenType.ID).text

            self.match(TokenType.ON)
            table = self.match(TokenType.ID).text

            self.match(TokenType.LPAREN)
            column = self.match(TokenType.ID).text
            self.match(TokenType.RPAREN)

            self.match(TokenType.USING)
            index_type = self.parse_index_type()

            self.match(TokenType.SEMICOLON)

            return CreateIndexStm(
                name=name,
                table=table,
                column=column,
                index_type=index_type
            )

        raise SyntaxError(
            f"Error sintactico: se esperaba TABLE o INDEX, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    def parse_create_table_body(self, name: str) -> CreateTableStm:
        if self.check(TokenType.LPAREN):
            self.match(TokenType.LPAREN)

            columns = self.parse_column_def_list()

            self.match(TokenType.RPAREN)

            storage = StorageType.HEAP

            if self.check(TokenType.USING):
                storage = self.parse_using_clause()

            csv = None

            if self.check(TokenType.FROM):
                csv = self.parse_csv_clause()

            self.match(TokenType.SEMICOLON)

            primary_key = None

            if columns is not None:
                for column in columns:
                    if column.primary_key:
                        primary_key = column.name
                        break

            return CreateTableStm(
                name=name,
                columns=columns,
                storage=storage,
                csv=csv,
                primary_key=primary_key
            )

        elif self.check(TokenType.FROM):
            csv = self.parse_csv_clause()

            storage = StorageType.HEAP

            if self.check(TokenType.USING):
                storage = self.parse_using_clause()

            primary_key = self.parse_primary_key_clause()

            self.match(TokenType.SEMICOLON)

            return CreateTableStm(
                name=name,
                columns=None,
                storage=storage,
                csv=csv,
                primary_key=primary_key
            )

            raise SyntaxError(
                f"Error sintactico: se esperaba '(' o FROM, "
                f"pero se encontro {self.current.type.name} "
                f"('{self.current.text}')"
            )

    def parse_column_def_list(self) -> list[ColumnDef]:
        columns: list[ColumnDef] = []

        columns.append(self.parse_column_def())

        while self.check(TokenType.COMMA):
            self.match(TokenType.COMMA)
            columns.append(self.parse_column_def())

        return columns

    def parse_column_def(self) -> ColumnDef:
        name = self.match(TokenType.ID).text
        data_type, size = self.parse_type()

        primary_key = False
        unique = False

        if self.check(TokenType.PRIMARY):
            self.match(TokenType.PRIMARY)
            self.match(TokenType.KEY)
            primary_key = True
            unique = True

        elif self.check(TokenType.UNIQUE):
            self.match(TokenType.UNIQUE)
            unique = True

        return ColumnDef(
            name=name,
            data_type=data_type,
            size=size,
            primary_key=primary_key,
            unique=unique
        )

    def parse_type(self) -> tuple[DataType, int | None]:
        if self.check(TokenType.INT):
            self.match(TokenType.INT)
            return DataType.INT, None

        elif self.check(TokenType.FLOAT):
            self.match(TokenType.FLOAT)
            return DataType.FLOAT, None

        elif self.check(TokenType.DOUBLE):
            self.match(TokenType.DOUBLE)
            return DataType.DOUBLE, None

        elif self.check(TokenType.CHAR):
            self.match(TokenType.CHAR)
            self.match(TokenType.LPAREN)

            size = int(self.match(TokenType.INT_LITERAL).text)

            self.match(TokenType.RPAREN)

            return DataType.CHAR, size

        raise SyntaxError(
            f"Error sintactico: se esperaba un tipo, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    def parse_using_clause(self) -> StorageType:
        self.match(TokenType.USING)
        return self.parse_storage_type()

    def parse_storage_type(self) -> StorageType:
        if self.check(TokenType.HEAP):
            self.match(TokenType.HEAP)
            return StorageType.HEAP
        elif self.check(TokenType.SEQUENTIAL):
            self.match(TokenType.SEQUENTIAL)
            return StorageType.SEQUENTIAL

        raise SyntaxError(
            f"Error sintactico: se esperaba HEAP o SEQUENTIAL, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    def parse_csv_clause(self) -> CsvSource:
        self.match(TokenType.FROM)
        self.match(TokenType.CSV)
        self.match(TokenType.LPAREN)

        path = self.match(TokenType.STRING_LITERAL).text
        delimiter = ","

        if self.check(TokenType.COMMA):
            self.match(TokenType.COMMA)
            self.match(TokenType.DELIMITER)
            self.match(TokenType.EQ)
            delimiter = self.match(TokenType.STRING_LITERAL).text

        self.match(TokenType.RPAREN)

        return CsvSource(path=path, delimiter=delimiter)

    def parse_primary_key_clause(self) -> str:
        self.match(TokenType.PRIMARY)
        self.match(TokenType.KEY)
        self.match(TokenType.LPAREN)

        column = self.match(TokenType.ID).text

        self.match(TokenType.RPAREN)

        return column

    def parse_index_type(self) -> IndexType:
        if self.check(TokenType.BTREE):
            self.match(TokenType.BTREE)
            return IndexType.BTREE
        elif self.check(TokenType.HASH):
            self.match(TokenType.HASH)
            return IndexType.HASH

        raise SyntaxError(
            f"Error sintactico: se esperaba BTREE o HASH, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    # INSERT

    def parse_insert_stmt_rest(self) -> InsertStm:
        self.match(TokenType.INTO)

        table = self.match(TokenType.ID).text

        self.match(TokenType.VALUES)
        self.match(TokenType.LPAREN)

        values = self.parse_value_list()

        self.match(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)

        return InsertStm(
            table=table,
            values=values
        )

    def parse_value_list(self) -> list[Exp]:
        values: list[Exp] = []

        values.append(self.parse_value())

        while self.check(TokenType.COMMA):
            self.match(TokenType.COMMA)
            values.append(self.parse_value())

        return values

    def parse_value(self) -> Exp:
        if self.check(TokenType.INT_LITERAL):
            value = int(self.match(TokenType.INT_LITERAL).text)
            return LiteralExp(value)

        elif self.check(TokenType.REAL_LITERAL):
            value = float(self.match(TokenType.REAL_LITERAL).text)
            return LiteralExp(value)

        elif self.check(TokenType.STRING_LITERAL):
            value = self.match(TokenType.STRING_LITERAL).text
            return LiteralExp(value)

        raise SyntaxError(
            f"Error sintactico: se esperaba un valor, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
    )

    # ALTER

    def parse_alter_stmt_rest(self) -> AlterStm:
        self.match(TokenType.TABLE)

        table = self.match(TokenType.ID).text

        self.match(TokenType.ADD)
        self.match(TokenType.UNIQUE)
        self.match(TokenType.LPAREN)

        column = self.match(TokenType.ID).text

        self.match(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)

        return AlterStm(
            table=table,
            column=column
        )

    # SELECT

    def parse_select_stmt_rest(self) -> SelectStm:
        columns = self.parse_select_list()

        self.match(TokenType.FROM)
        table = self.match(TokenType.ID).text

        force_indexes = None
        force_seqscan = False
        force_all_indexes = False
        if self.check(TokenType.FORCE):
            force_indexes, force_seqscan, force_all_indexes = self.parse_force_clause()

        where = None
        if self.check(TokenType.WHERE):
            where = self.parse_where_clause()

        self.match(TokenType.SEMICOLON)

        return SelectStm(
            columns=columns,
            table=table,
            force_indexes=force_indexes,
            force_seqscan=force_seqscan,
            force_all_indexes=force_all_indexes,
            where=where
        )

    def parse_select_list(self) -> list[str] | None:
        if self.check(TokenType.STAR):
            self.match(TokenType.STAR)
            return None

        return self.parse_column_list()

    def parse_column_list(self) -> list[str]:
        columns: list[str] = []

        columns.append(self.match(TokenType.ID).text)

        while self.check(TokenType.COMMA):
            self.match(TokenType.COMMA)
            columns.append(self.match(TokenType.ID).text)

        return columns

    def parse_force_clause(self) -> tuple[list[str] | None, bool, bool]:
        self.match(TokenType.FORCE)

        if self.check(TokenType.SEQSCAN):
            self.match(TokenType.SEQSCAN)
            return None, True, False

        self.match(TokenType.INDEX)
        self.match(TokenType.LPAREN)

        if self.check(TokenType.STAR):
            self.match(TokenType.STAR)
            self.match(TokenType.RPAREN)
            return None, False, True

        indexes = self.parse_index_list()

        self.match(TokenType.RPAREN)

        return indexes, False, False

    def parse_index_list(self) -> list[str]:
        indexes: list[str] = []

        indexes.append(self.match(TokenType.ID).text)

        while self.check(TokenType.COMMA):
            self.match(TokenType.COMMA)
            indexes.append(self.match(TokenType.ID).text)

        return indexes

    # WHERE

    def parse_where_clause(self) -> Exp:
        self.match(TokenType.WHERE)
        return self.parse_condition()

    def parse_condition(self) -> Exp:
        return self.parse_or_condition()

    def parse_or_condition(self) -> Exp:
        expression = self.parse_and_condition()

        while self.check(TokenType.OR):
            self.match(TokenType.OR)
            right = self.parse_and_condition()
            expression = BinaryExp(expression, right, BinaryOp.OR)

        return expression

    def parse_and_condition(self) -> Exp:
        expression = self.parse_not_condition()

        while self.check(TokenType.AND):
            self.match(TokenType.AND)
            right = self.parse_not_condition()
            expression = BinaryExp(expression, right, BinaryOp.AND)

        return expression
  

    def parse_not_condition(self) -> Exp:
        if self.check(TokenType.NOT):
            self.match(TokenType.NOT)
            return UnaryExp(self.parse_not_condition(), UnaryOp.NOT)

        return self.parse_primary_condition()

    def parse_primary_condition(self) -> Exp:
        if self.check(TokenType.LPAREN):
            self.match(TokenType.LPAREN)
            expression = self.parse_condition()
            self.match(TokenType.RPAREN)
            return expression

        left = self.parse_operand()
        op = self.parse_comparison_operator()
        right = self.parse_operand()

        return BinaryExp(left, right, op)

    def parse_operand(self) -> Exp:
        if self.check(TokenType.ID):
            return IdExp(self.match(TokenType.ID).text)

        if self.check(TokenType.INT_LITERAL):
            return LiteralExp(int(self.match(TokenType.INT_LITERAL).text))

        if self.check(TokenType.REAL_LITERAL):
            return LiteralExp(float(self.match(TokenType.REAL_LITERAL).text))

        if self.check(TokenType.STRING_LITERAL):
            return LiteralExp(self.match(TokenType.STRING_LITERAL).text)

        raise SyntaxError(
            f"Error sintactico: se esperaba un operando, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    def parse_comparison(self) -> Exp:
        left = self.parse_operand()
        operator = self.parse_comparison_operator()
        right = self.parse_operand()

        return BinaryExp(left, right, operator)

    def parse_comparison_operator(self) -> BinaryOp:
        if self.check(TokenType.EQ):
            self.match(TokenType.EQ)
            return BinaryOp.EQ

        if self.check(TokenType.NEQ):
            self.match(TokenType.NEQ)
            return BinaryOp.NEQ

        if self.check(TokenType.NEQ2):
            self.match(TokenType.NEQ2)
            return BinaryOp.NEQ2

        if self.check(TokenType.LT):
            self.match(TokenType.LT)
            return BinaryOp.LT

        if self.check(TokenType.LE):
            self.match(TokenType.LE)
            return BinaryOp.LE

        if self.check(TokenType.GT):
            self.match(TokenType.GT)
            return BinaryOp.GT

        if self.check(TokenType.GE):
            self.match(TokenType.GE)
            return BinaryOp.GE

        raise SyntaxError(
            f"Error sintactico: se esperaba un operador de comparacion, "
            f"pero se encontro {self.current.type.name} "
            f"('{self.current.text}')"
        )

    # DELETE

    def parse_delete_stmt_rest(self) -> DeleteStm:
        self.match(TokenType.FROM)

        table = self.match(TokenType.ID).text

        force_indexes = None
        force_seqscan = False
        force_all_indexes = False
        if self.check(TokenType.FORCE):
            force_indexes, force_seqscan, force_all_indexes = self.parse_force_clause()

        where = None

        if self.check(TokenType.WHERE):
            where = self.parse_where_clause()

        self.match(TokenType.SEMICOLON)

        return DeleteStm(
            table=table,
            force_indexes=force_indexes,
            force_seqscan=force_seqscan,
            force_all_indexes=force_all_indexes,
            where=where
        )