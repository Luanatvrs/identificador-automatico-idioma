import pandas as pd
import random
import re
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ingles = [
"Hello, how are you?",
"I love to read books.",
"The weather is nice today.",
"Where is the nearest restaurant?",
"What time is it?",
"I enjoy playing soccer.",
"Can you help me with this?",
"I'm going to the movies tonight.",
"This is a beautiful place.",
"I like listening to music.",
"Do you speak English?",
"What is your favorite color?",
"I'm learning to play the guitar.",
"Have a great day!",
"I need to buy some groceries.",
"Let's go for a walk.",
"How was your weekend?",
"I'm excited for the concert.",
"Could you pass me the salt, please?",
"I have a meeting at 2 PM.",
"I'm planning a vacation.",
"She sings beautifully.",
"The cat is sleeping.",
"I want to learn French.",
"I enjoy going to the beach.",
"Where can I find a taxi?",
"I'm sorry for the inconvenience.",
"I'm studying for my exams.",
"I like to cook dinner at home.",
"Do you have any recommendations for restaurants?",
]

espanhol = [
"Hola, ¿cómo estás?",
"Me encanta leer libros.",
"El clima está agradable hoy.",
"¿Dónde está el restaurante más cercano?",
"¿Qué hora es?",
"Voy al parque todos los días.",
"¿Puedes ayudarme con esto?",
"Me gustaría ir de vacaciones.",
"Este es mi libro favorito.",
"Me gusta bailar salsa.",
"¿Hablas español?",
"¿Cuál es tu comida favorita?",
"Estoy aprendiendo a tocar el piano.",
"¡Que tengas un buen día!",
"Necesito comprar algunas frutas.",
"Vamos a dar un paseo.",
"¿Cómo estuvo tu fin de semana?",
"Estoy emocionado por el concierto.",
"¿Me pasas la sal, por favor?",
"Tengo una reunión a las 2 PM.",
"Estoy planeando unas vacaciones.",
"Ella canta hermosamente.",
"El perro está jugando.",
"Quiero aprender italiano.",
"Disfruto ir a la playa.",
"¿Dónde puedo encontrar un taxi?",
"Lamento las molestias.",
"Estoy estudiando para mis exámenes.",
"Me gusta cocinar la cena en casa.",
"¿Tienes alguna recomendación de restaurantes?",
]

portugues = [
"Estou indo para o trabalho agora.",
"Adoro passar tempo com minha família.",
"Preciso comprar leite e pão.",
"Vamos ao cinema no sábado.",
"Gosto de praticar esportes ao ar livre.",
"O trânsito está terrível hoje.",
"A comida estava deliciosa!",
"Você já visitou o Rio de Janeiro?",
"Tenho uma reunião importante amanhã.",
"A festa começa às 20h.",
"Estou cansado depois de um longo dia de trabalho.",
"Vamos fazer um churrasco no final de semana.",
"O livro que estou lendo é muito interessante.",
"Estou aprendendo a cozinhar pratos novos.",
"Preciso fazer exercícios físicos regularmente.",
"Vou viajar para o exterior nas férias.",
"Você gosta de dançar?",
"Hoje é meu aniversário!",
"Gosto de ouvir música clássica.",
"Estou estudando para o vestibular.",
"Meu time de futebol favorito ganhou o jogo.",
"Quero aprender a tocar violão.",
"Vamos fazer uma viagem de carro.",
"O parque fica cheio aos finais de semana.",
"O filme que assisti ontem foi ótimo.",
"Preciso resolver esse problema o mais rápido possível.",
"Adoro explorar novos lugares.",
"Vou visitar meus avós no domingo.",
"Estou ansioso para as férias de verão.",
"Gosto de fazer caminhadas na natureza.",
"O restaurante tem uma vista incrível.",
"Vamos sair para jantar no sábado.",
]

pre_padroes = []
for frase in ingles:
  pre_padroes.append( [frase, 'inglês'])

for frase in espanhol:
  pre_padroes.append( [frase, 'espanhol'])

for frase in portugues:
  pre_padroes.append( [frase, 'português'])

random.shuffle(pre_padroes)
print(pre_padroes)


dados = pd.DataFrame(pre_padroes)
dados

def tamanhoMedioFrases(texto):
  palavras = re.split("\s",texto)
  tamanhos = [len(s) for s in palavras if len(s)>0]
  soma = 0
  for t in tamanhos:
    soma=soma+t
  return soma / len(tamanhos)

# Conta acentos únicos pra pt e es
def quantidadeAcentos(frase, acentos):
    return float(sum(frase.lower().count(c) for c in acentos))

# Frequência de letras presentes no texto
def freqLetras(texto):
    texto = texto.lower()
    texto = re.sub('[^a-z]', '', texto)
    contagem = Counter(texto)
    total = sum(contagem.values())
    if total == 0:
        return [0]*26
    return [contagem.get(chr(c), 0)/total for c in range(ord('a'), ord('z')+1)]

palavras_pt = set([
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com',
    'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como',
    'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à'
])

palavras_es = set([
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por',
    'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero',
    'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque'
])

palavras_en = set([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for',
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his',
    'by', 'from', 'they', 'we', 'say', 'her', 'she'
])

palavras_chave = list(palavras_pt | palavras_es | palavras_en)

#presença de palavras chaves em cada idioma
def presencaPalvrasChave(texto, palavrasChave):
    texto = texto.lower()
    presentes = [1.0 if palavra in texto else 0.0 for palavra in palavrasChave]
    return presentes

def extraiCaracteristicas(frase):
    texto = frase[0]
    pattern_regex = re.compile(' [^\w+]', re.UNICODE)
    texto = re.sub(pattern_regex,' ',texto)

    qtdAcentosPr = quantidadeAcentos(texto, 'ãâç')
    qtdAcentosEs = quantidadeAcentos(texto, 'ñ')
    tamMedio = tamanhoMedioFrases(texto)
    frequenciaLetras = freqLetras(texto)
    presencaPalavrasChave = presencaPalvrasChave(texto, palavras_chave)

    padrao = [tamMedio] + frequenciaLetras + [qtdAcentosPr] + [qtdAcentosEs] + presencaPalavrasChave + [frase[1]]
    return padrao

def geraPadroes(frases):
  padroes = []
  for frase in frases:
    padrao = extraiCaracteristicas(frase)
    padroes.append(padrao)
  return padroes

padroes = geraPadroes(pre_padroes)

print(padroes)

dados = pd.DataFrame(padroes)
dados

vet = np.array(padroes)
classes = vet[:,-1]          
padroes_sem_classe = vet[:,0:-1]
X_train, X_test, y_train, y_test = train_test_split(padroes_sem_classe, classes, test_size=0.30, random_state=42, stratify=classes)

treinador = svm.SVC() 
modelo = treinador.fit(X_train, y_train)

# score com os dados de treinamento
acuracia = modelo.score(X_train, y_train)
print("Acurácia nos dados de treinamento: {:.2f}%".format(acuracia * 100))

# com dados de teste que não foram usados no treinamento
print('matriz de confusão')
y_pred2 = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred2)
print(cm)
print(classification_report(y_test, y_pred2))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Dados de teste tratados (y_test, y_pred2)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred2, cmap='Blues')
plt.title('Matriz de Confusão - Dados de Teste')
plt.show()