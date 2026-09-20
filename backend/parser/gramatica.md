# Gramática EBNF LL(1)

`{}` = repetición
`[]` = opcional

```ebnf
Program ::= { Statement }


Statement ::= CREATE CreateStmtRest
            | INSERT InsertStmtRest
            | ALTER AlterStmtRest
            | SELECT SelectStmtRest
            | DELETE DeleteStmtRest


CreateStmtRest ::= TABLE ID CreateTableBody
                 | INDEX ID ON ID "(" ID ")"
                   USING IndexType ";"


CreateTableBody ::= "(" ColumnDefList ")"
                    [ UsingClause ]
                    [ CsvClause ]
                    ";"

                  | CsvClause
                    [ UsingClause ]
                    PrimaryKeyClause
                    ";"


ColumnDefList ::= ColumnDef { "," ColumnDef }

ColumnDef ::= ID Type [ ColumnConstraint ]

ColumnConstraint ::= PRIMARY KEY
                   | UNIQUE


Type ::= INT
       | FLOAT
       | DOUBLE
       | CHAR "(" INT_LITERAL ")"


UsingClause ::= USING StorageType

StorageType ::= HEAP
              | SEQUENTIAL


CsvClause ::= FROM CSV "(" STRING_LITERAL
              [ "," DELIMITER "=" STRING_LITERAL ]
              ")"


PrimaryKeyClause ::= PRIMARY KEY "(" ID ")"


InsertStmtRest ::= INTO ID
                   VALUES "(" ValueList ")"
                   ";"

ValueList ::= Value { "," Value }

Value ::= INT_LITERAL
        | REAL_LITERAL
        | STRING_LITERAL


IndexType ::= BTREE
            | HASH


AlterStmtRest ::= TABLE ID
                  ADD UNIQUE "(" ID ")"
                  ";"


SelectStmtRest ::= SelectList
                   FROM ID
                   [ ForceClause ]
                   [ WhereClause ]
                   ";"

SelectList ::= "*"
             | ColumnList

ColumnList ::= ID { "," ID }


ForceClause ::= FORCE ForceClauseRest

ForceClauseRest ::= INDEX "(" IndexBody ")"
                  | SEQSCAN

IndexBody ::= "*"
            | IndexList

IndexList ::= ID { "," ID }


WhereClause ::= WHERE Condition

Condition ::= OrCondition

OrCondition ::= AndCondition
                { OR AndCondition }

AndCondition ::= NotCondition
                 { AND NotCondition }

NotCondition ::= NOT NotCondition
               | PrimaryCondition

PrimaryCondition ::= "(" Condition ")"
                   | Operand ComparisonOperator Operand


Operand ::= ID
          | INT_LITERAL
          | REAL_LITERAL
          | STRING_LITERAL


ComparisonOperator ::= "="
                     | "!="
                     | "<>"
                     | "<"
                     | "<="
                     | ">"
                     | ">="


DeleteStmtRest ::= FROM ID
                   [ ForceClause ]
                   [ WhereClause ]
                   ";"
```

