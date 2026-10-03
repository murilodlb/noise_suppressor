# Noise Suppressor

Sistema de supressão de ruído não-estacionário usando Redes Neurais Convolucionais (U-Net) e Processamento Digital de Sinais (PDS), focado em viabilidade para futura implantação em **Edge Computing / TinyML**.

## A Ideia

Filtros adaptativos clássicos (como LMS e Wiener) falham ao lidar com ruídos abruptos e não-estacionários devido ao seu tempo de convergência. Para contornar isso, este projeto adota um paradigma de visão computacional para o áudio:

1. O sinal de voz é convertido em um **Espectrograma** (matriz 2D tempo-frequência) via STFT.
2. Uma rede neural **TinyU-Net** atua como um filtro espacial inteligente, analisando as texturas do espectrograma para prever uma **Máscara de Proporção Ideal (IRM)**.
3. A máscara é multiplicada pelo espectrograma ruidoso, atenuando as frequências do ruído e preservando a voz.
4. O sinal no domínio do tempo é reconstruído via iSTFT com o algoritmo Overlap-Add.

Como a rede prevê apenas a máscara de magnitude (valores contínuos entre 0 e 1), a topologia se mantém leve e otimizada para restrições severas de memória e processamento em microcontroladores.

## Resultados da Prova de Conceito

Abaixo, a validação visual do fluxo de supressão aplicando a máscara sobre o espectrograma ruidoso e reconstruindo um sinal desconhecido pelo treinamento.

### Domínio do Tempo

Comparação entre o sinal original limpo, o sinal corrompido por ruído e o sinal reconstruído após a filtragem espacial pela rede.

![Sinais no Domínio do Tempo](data/imgs/time_domain.png)

### Domínio da Frequência

Comparação das matrizes de magnitude em escala de Decibéis (dB). Nota-se a atenuação do ruído de fundo (áreas escuras) no espectrograma filtrado, preservando a estrutura harmônica da voz.

![Espectrogramas](data/imgs/spectrograms.png)

## Arquitetura do Projeto

```text
noise_supressor/
├── data/
│   ├── clean/               # Amostras de voz limpa (Mozilla Common Voice)
│   ├── noise/               # Amostras de ruído ambiental
│   ├── precomputed/         # Tensores .pt (noisy_mag, IRM) pré-computados
│   ├── test/                # Amostras de áudio para validação
│   ├── audio/               # Áudios gerados pelo demo (clean, noisy, enhanced)
│   └── imgs/                # Gráficos gerados pelo demo
├── src/
│   ├── cli.py               # Ponto de entrada unificado da CLI
│   ├── inference.py         # Módulo compartilhado de inferência e aprimoramento
│   ├── metrics.py           # Métricas de PDS, classificação T-F e inteligibilidade
│   ├── evaluate.py          # Avaliação em holdout vs Noisy e Oracle IRM
│   ├── benchmark.py         # Medição de overhead (MACs, memória, latência, RTF)
│   ├── dsp_utils.py         # Funções de PDS (STFT, iSTFT, IRM, Plots)
│   ├── model.py             # Arquitetura minimalista da TinyU-Net
│   ├── dataset_builder.py   # Carregamento e geração de tensores
│   ├── preprocess_dataset.py# Pré-cálculo do dataset em .pt
│   ├── train.py             # Loop de treinamento PyTorch com prevenção de leakage
│   └── main.py              # Script da Prova de Conceito (PoC)
├── checkpoints/             # Pesos treinados (.pth)
├── reports/                 # Relatórios de avaliação, benchmark e inferência
├── pyproject.toml           # Configuração de projeto e dependências via uv
└── uv.lock                  # Lockfile reproduzível do uv
```

## Instalação e Ambiente (`uv`)

Este projeto utiliza o [`uv`](https://docs.astral.sh/uv/) para gerenciamento de dependências e ambientes virtuais ultrarrápidos e reproduzíveis.

```bash
# 1. Instalar dependências e criar o ambiente virtual
uv sync

# 2. Executar a suíte de testes unitários
uv run pytest
```

## Pipeline de Execução (CLI)

O projeto expõe o comando `noise_suppressor` gerenciado pelo `uv`:

### 1. Pré-processamento do Dataset (I/O Optimization)
Pré-calcula as magnitudes STFT e as máscaras IRM para disco em tensores `.pt`, acelerando drasticamente o treino:
```bash
uv run noise_suppressor preprocess --samples_per_clean 1
```

### 2. Treinamento da TinyU-Net
Treina o modelo prevenindo vazamento de dados (*leakage* por locutor) via divisão dos arquivos de voz de origem:
```bash
uv run noise_suppressor train --epochs 20 --batch_size 4 --learning_rate 1e-3
```

### 3. Inferência em Áudios Arbitrários
Processa um ou mais arquivos de áudio ruidosos (suporta qualquer duração e formato `.wav`, `.mp3`, `.webm`):
```bash
uv run noise_suppressor infer --input data/audio/noisy.wav --output_dir reports/inference
```

### 4. Demonstração Interativa (PoC)
Executa a demonstração ponta a ponta com áudio de teste e gera os gráficos comparativos em `data/imgs/`:
```bash
uv run noise_suppressor demo
```

### 5. Avaliação de Qualidade de Sinal (Signal Quality Evaluation)
Avalia o modelo treinado sobre o split de validação isolado contra o sinal ruidoso (*baseline*) e a máscara ideal (*oracle IRM*):
```bash
uv run noise_suppressor evaluate --max_samples 30 --lc_db 0.0
```
Gera relatórios detalhados em `reports/evaluation/<timestamp>/`:
- `summary.md`: Tabelas em Markdown com comparações e agrupamento por faixas de SNR.
- `summary.json`: Todas as médias, medianas e desvios padrão agregados.
- `per_sample.csv`: Métricas individuais de cada áudio avaliado.

### 6. Benchmark de Overhead Computacional (TinyML / Edge)
Mede o custo computacional, footprint de memória e velocidade de processamento:
```bash
uv run noise_suppressor benchmark --seconds 2.0 --repeats 50
```
Gera o relatório em `reports/benchmark/benchmark.json`.

---

## Métricas de Avaliação de Sinal

Para avaliar sistemas de processamento de sinais de forma análoga a classificadores tradicionais de IA (Precision, Recall, F1), o pipeline projeta a tarefa no domínio tempo-frequência (T-F):

| Conceito Clássico | Equivalente no PDS | Significado |
|---|---|---|
| **Precision** | **T-F Bin Precision** | Dos bins de frequência onde a rede manteve sinal, quantos eram realmente voz? Evita passagem de ruído residual. |
| **Recall (Hit Rate)** | **T-F Bin Recall** | Dos bins onde havia voz original, quantos a máscara preservou? Evita corte da voz do locutor. |
| **F1-Score** | **T-F Mask F1** | Média harmônica entre preservação da voz e eliminação do ruído no espectrograma. |
| **HIT − FA** | **Hit Rate − False Alarm** | Métrica canônica da literatura de separação de fala (Kim et al., 2009). |
| **Speech Attenuation (SA)** | $10 \log_{10} \frac{\|S\|^2}{\|M \odot S\|^2}$ | Perda de energia da fala em dB (quanto menor, melhor). |
| **Noise Attenuation (NA)** | $10 \log_{10} \frac{\|N\|^2}{\|M \odot N\|^2}$ | Atenuação do ruído em dB (quanto maior, melhor). |
| **SI-SDR** | Scale-Invariant SDR (dB) | Fidelidade da onda no domínio do tempo independente de ganho de escala. |
| **PESQ** | ITU-T P.862.2 (Wideband) | Qualidade perceptual percebida pelo ouvido humano (escala 1.0 a 4.5). |
| **STOI / ESTOI** | Short-Time Objective Intelligibility | Inteligibilidade objetiva da fala (escala 0.0 a 1.0). |

---

## Avaliação de Overhead Computacional

Resultados obtidos no benchmark com a `TinyUNet` (áudio de 2.0 segundos, 16 kHz):

- **Parâmetros Totais:** ~29.641 parâmetros
- **Tamanho do Modelo:** ~117 KB (Float32) / ~29 KB (Quantizado Int8 estimado)
- **Operações:** ~98 MMACs/segundo de áudio
- **Latência Total por Clip (CPU):** ~14.5 ms (Forward da rede: ~11.5 ms; STFT + iSTFT + Pré-processamento: ~3.0 ms)
- **Real-Time Factor (RTF):** ~0.0073 (Processa **137x mais rápido** que tempo real em CPU comum)
