from token import Token, TokenType


class Scanner:
    def __init__(self, source: str):
        self.input = source
        self.first = 0
        self.current = 0

    def is_white_space(self, c: str) -> bool:
        return c in " \n\r\t"

    def nextToken(self) -> Token:
        # Saltar espacios
        while self.current < len(self.input) and \
                self.is_white_space(self.input[self.current]):
            self.current += 1

        # Fin de entrada
        if self.current >= len(self.input):
            return Token(TokenType.EOF)

        self.first = self.current
        c = self.input[self.current]

        # Identificadores y palabras reservadas
        if c.isalpha() or c == "_":
            self.current += 1

            while self.current < len(self.input):
                c = self.input[self.current]

                if c.isalnum() or c == "_":
                    self.current += 1
                else:
                    break

            lexema = self.input[self.first:self.current]
            upper = lexema.upper()

            if upper == "CREATE":
                return Token(TokenType.CREATE, lexema)
            elif upper == "TABLE":
                return Token(TokenType.TABLE, lexema)
            elif upper == "INDEX":
                return Token(TokenType.INDEX, lexema)
            elif upper == "ON":
                return Token(TokenType.ON, lexema)
            elif upper == "INSERT":
                return Token(TokenType.INSERT, lexema)
            elif upper == "INTO":
                return Token(TokenType.INTO, lexema)
            elif upper == "VALUES":
                return Token(TokenType.VALUES, lexema)
            elif upper == "DELETE":
                return Token(TokenType.DELETE, lexema)
            elif upper == "ALTER":
                return Token(TokenType.ALTER, lexema)
            elif upper == "ADD":
                return Token(TokenType.ADD, lexema)
            elif upper == "UNIQUE":
                return Token(TokenType.UNIQUE, lexema)
            elif upper == "SELECT":
                return Token(TokenType.SELECT, lexema)
            elif upper == "FROM":
                return Token(TokenType.FROM, lexema)
            elif upper == "WHERE":
                return Token(TokenType.WHERE, lexema)
            elif upper == "FORCE":
                return Token(TokenType.FORCE, lexema)
            elif upper == "SEQSCAN":
                return Token(TokenType.SEQSCAN, lexema)
            elif upper == "USING":
                return Token(TokenType.USING, lexema)
            elif upper == "HEAP":
                return Token(TokenType.HEAP, lexema)
            elif upper == "SEQUENTIAL":
                return Token(TokenType.SEQUENTIAL, lexema)
            elif upper == "CSV":
                return Token(TokenType.CSV, lexema)
            elif upper == "DELIMITER":
                return Token(TokenType.DELIMITER, lexema)
            elif upper == "PRIMARY":
                return Token(TokenType.PRIMARY, lexema)
            elif upper == "KEY":
                return Token(TokenType.KEY, lexema)
            elif upper == "BTREE":
                return Token(TokenType.BTREE, lexema)
            elif upper == "HASH":
                return Token(TokenType.HASH, lexema)
            elif upper == "AND":
                return Token(TokenType.AND, lexema)
            elif upper == "OR":
                return Token(TokenType.OR, lexema)
            elif upper == "NOT":
                return Token(TokenType.NOT, lexema)
            elif upper == "INT":
                return Token(TokenType.INT, lexema)
            elif upper == "FLOAT":
                return Token(TokenType.FLOAT, lexema)
            elif upper == "DOUBLE":
                return Token(TokenType.DOUBLE, lexema)
            elif upper == "CHAR":
                return Token(TokenType.CHAR, lexema)
            else:
                return Token(TokenType.ID, lexema)

        # Numeros
        elif c.isdigit():
            self.current += 1

            while self.current < len(self.input) and \
                    self.input[self.current].isdigit():
                self.current += 1

            # Parte decimal
            if self.current < len(self.input) and \
                    self.input[self.current] == ".":

                if self.current + 1 < len(self.input) and \
                        self.input[self.current + 1].isdigit():

                    self.current += 1

                    while self.current < len(self.input) and \
                            self.input[self.current].isdigit():
                        self.current += 1

                    text = self.input[self.first:self.current]
                    return Token(TokenType.REAL_LITERAL, text)

            text = self.input[self.first:self.current]
            return Token(TokenType.INT_LITERAL, text)

        # Strings
        elif c == "'":
            self.current += 1

            while self.current < len(self.input) and \
                    self.input[self.current] != "'":
                self.current += 1

            if self.current >= len(self.input):
                text = self.input[self.first:self.current]
                return Token(TokenType.ERR, text)

            self.current += 1
            text = self.input[self.first:self.current]

            return Token(TokenType.STRING_LITERAL, text)

        # Operadores y simbolos
        elif c == "(":
            self.current += 1
            return Token(TokenType.LPAREN, c)

        elif c == ")":
            self.current += 1
            return Token(TokenType.RPAREN, c)

        elif c == ",":
            self.current += 1
            return Token(TokenType.COMMA, c)

        elif c == ";":
            self.current += 1
            return Token(TokenType.SEMICOLON, c)

        elif c == "*":
            self.current += 1
            return Token(TokenType.STAR, c)

        elif c == "=":
            self.current += 1
            return Token(TokenType.EQ, c)

        elif c == "!":
            if self.current + 1 < len(self.input) and \
                    self.input[self.current + 1] == "=":
                self.current += 2
                return Token(TokenType.NEQ, "!=")

            self.current += 1
            return Token(TokenType.ERR, c)

        elif c == "<":
            if self.current + 1 < len(self.input):
                next_c = self.input[self.current + 1]

                if next_c == "=":
                    self.current += 2
                    return Token(TokenType.LE, "<=")

                elif next_c == ">":
                    self.current += 2
                    return Token(TokenType.NEQ2, "<>")

            self.current += 1
            return Token(TokenType.LT, "<")

        elif c == ">":
            if self.current + 1 < len(self.input) and \
                    self.input[self.current + 1] == "=":
                self.current += 2
                return Token(TokenType.GE, ">=")

            self.current += 1
            return Token(TokenType.GT, ">")

        # Caracter invalido
        else:
            self.current += 1
            return Token(TokenType.ERR, c)


def ejecutar_scanner(scanner: Scanner):
    while True:
        token = scanner.nextToken()
        print(token)

        if token.type == TokenType.EOF:
            print("\nScanner exitoso")
            return

        if token.type == TokenType.ERR:
            print("\nScanner no exitoso")
            return