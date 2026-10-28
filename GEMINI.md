# GEMINI.md: Contexto do Projeto de Supressão de Ruído

O projeto consiste em um sistema de supressão de áudio não-estacionário que integra Processamento Digital de Sinais (PDS) e técnicas de Visão Computacional, com sua árvore de código organizada sob o diretório `src`.

## Componentes Técnicos e Arquitetura

* **Transformações de Domínio (`dsp_utils.py`):** O sistema transita o áudio do domínio do tempo para o domínio da frequência utilizando a Transformada de Fourier de Curto Termo (STFT). A reconstrução do áudio filtrado de volta ao domínio do tempo é feita pela Transformada Inversa (iSTFT) utilizando o algoritmo Overlap-Add.


* **Cálculo de Máscaras e Estabilidade:** A Máscara de Proporção Ideal (IRM) é gerada computando a relação de energia ao quadrado entre o sinal limpo e o ruído, com uma restrição matemática que garante valores contínuos entre 0 e 1.


* **Tratamento de Dados (`dataset_builder.py`):** O espectrograma de entrada não é alimentado de forma linear na rede; ele passa por compressão logarítmica (`np.log1p`) e normalização Z-Score para estabilizar a aprendizagem do modelo. Além disso, o pipeline suporta a pré-computação do dataset gerando arquivos `.pt` para agilizar o tempo de leitura no treinamento.


* **Arquitetura da Rede (`model.py`):** O núcleo da inteligência, por enquanto, é a `TinyUNet`, uma arquitetura convolucional minimalista desenhada para cenários de TinyML. Ela emprega blocos de convolução dupla em caminhos de *downsampling* (Encoder) e *upsampling* (Decoder), além de usar interpolação no final para garantir correspondência exata de dimensões espaciais.


* **Estratégia de Treinamento (`train.py`):** O treinamento previne o vazamento de dados (data leakage) ao realizar o *split* de validação isolando os arquivos limpos de origem através da função `_split_indices_by_clean_source`. O loop utiliza a função de perda `BCELoss` e ajusta a taxa de aprendizado dinamicamente usando o agendador `ReduceLROnPlateau`.


* **Orquestração e Validação (`main.py`):** A Prova de Conceito (PoC) testa a inferência injetando ruído em uma amostra isolada (holdout), aplicando a máscara gerada pela U-Net, e exportando tanto os arquivos `.wav` reconstruídos quanto os espectrogramas e ondas para análise visual em disco.

* **Objetivo:** Pretende-se usar esse modelo embarcado, uma possível forma de fazer isso é treinar o modelo com um HPC usando GPU e salvar os pesos para serem utilizados no sistema embarcado como um ESP32, por exemplo. Atualmente o projeto se encontra na fase de pesquisa, com a TinyUNet sendo o primeiro candidato a modelo escolhido. Se o modelo funcionar bem em um computador padrão, o projeto segue para a etapa de TinyML com a pesquisa de viabilidade de desempenho em sistemas modestos embarcados.

## Regras

* Todas as variáveis, funções, classes, tudo deve ser escrito em inglês usando snake case.


