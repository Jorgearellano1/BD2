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
            | SET SetStmtRest


CreateStmtRest ::= TABLE ID CreateTableBody
                 | INDEX ID ON ID "(" IDList ")"
                   USING IndexType
                   [ "WITH" "(" OptionList ")" ]
                   ";"


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
       | VECTOR "(" INT_LITERAL ")"


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
            | RTREE
            | INVERTED
            | IVF


OptionList ::= Option { "," Option }

Option ::= ID "=" ( INT_LITERAL | STRING_LITERAL )


AlterStmtRest ::= TABLE ID
                  ADD UNIQUE "(" ID ")"
                  ";"


SelectStmtRest ::= SelectList
                   FROM ID
                   [ ForceClause ]
                   [ WhereClause ]
                   [ OrderClause ]
                   [ LimitClause ]
                   ";"

SelectList ::= "*"
             | SelectItem { "," SelectItem }

SelectItem ::= ID
             | COSINE_SCORE "(" ")" [ AS ID ]
             | ID DistanceOperator STRING_LITERAL [ AS ID ]

DistanceOperator ::= "<->"
                   | "<=>"
                   | "<#>"


ForceClause ::= FORCE ForceClauseRest

ForceClauseRest ::= INDEX "(" IndexBody ")"
                  | SEQSCAN

IndexBody ::= "*"
            | IndexList

IndexList ::= ID { "," ID }

IDList ::= ID { "," ID }


WhereClause ::= WHERE Condition

Condition ::= OrCondition

OrCondition ::= AndCondition
                { OR AndCondition }

AndCondition ::= NotCondition
                 { AND NotCondition }

NotCondition ::= NOT NotCondition
               | PrimaryCondition

PrimaryCondition ::= "(" Condition ")"
                   | ID MATCH STRING_LITERAL
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


OrderClause ::= ORDER BY ID [ ASC | DESC ]

LimitClause ::= LIMIT INT_LITERAL


SetStmtRest ::= ID [ "." ID ] "=" ( INT_LITERAL | REAL_LITERAL | STRING_LITERAL ) ";"


DeleteStmtRest ::= FROM ID
                   [ ForceClause ]
                   [ WhereClause ]
                   ";"
```
