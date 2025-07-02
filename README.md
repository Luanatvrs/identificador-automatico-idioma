# Language Classification using SVM

Projeto para classificar frases em três idiomas: inglês, espanhol e português, utilizando aprendizado de máquina com SVM.

As frases são pré-processadas e extraídas características linguísticas, como tamanho médio das palavras, frequência de letras, contagem de acentos e presença de palavras-chave específicas de cada idioma. O modelo SVM é treinado para identificar o idioma da frase.

## Dados

Frases de exemplo em três idiomas:

- Inglês (30 frases)  
- Espanhol (30 frases)  
- Português (32 frases)  

## Técnicas Aplicadas

- Tamanho médio das palavras  
- Frequência das letras (a-z)  
- Quantidade de acentos específicos (ã, â, ç para português; ñ para espanhol)  
- Presença de palavras-chave comuns a cada idioma  

## Requisitos

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

## Execução

1. Instale as dependências:

```bash
pip install pandas numpy scikit-learn matplotlib
```
2. Execute:

```bash
python main.py
```
## Estrutura

```
├── main.py               # Código do projeto
```

## Resultados

| Classe     | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| espanhol   | 1.00      | 0.33   | 0.50     | 9       |
| inglês     | 0.89      | 0.89   | 0.89     | 9       |
| português  | 0.62      | 1.00   | 0.77     | 10      |

**Accuracy:** 0.75 (28 amostras)  
**Macro avg:** Precision 0.84 | Recall 0.74 | F1-score 0.72  
**Weighted avg:** Precision 0.83 | Recall 0.75 | F1-score 0.72  

## Autora

Luana – Desenvolvedora em IA aplicada e otimização de modelos.
