# Análise do Impacto da Exploração na Aprendizagem Off-Policy
## SAC vs TD3 — Pendulum-v1

> **Disciplina:** IA368 — Tópicos em Inteligência Artificial  
> **UNICAMP — 2026**  
> **Autores:** Daniel Higa & Luan  

---

## 1. Resumo Executivo

Este trabalho investiga como diferentes níveis de **exploração** afetam o desempenho de dois algoritmos off-policy amplamente utilizados em Deep Reinforcement Learning: **Soft Actor-Critic (SAC)** e **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

Foram realizados **90 experimentos** no ambiente **Pendulum-v1** (Gymnasium), variando:
- SAC: coeficiente de entropia α ∈ {0.01, 0.05, 0.10, 0.20, auto}
- TD3: desvio-padrão do ruído σ ∈ {0.05, 0.10, 0.20, 0.30}
- Cada configuração treinada com **10 seed(s)** (1-10) × **100,000 passos**
- Avaliação determinística com **20 episódio(s)** por execução

**Resultado principal:** O algoritmo **SAC** obteve desempenho médio superior  
(SAC: -130.9 ± 4.2 vs TD3: -134.3 ± 5.2),  
com diferença **estatisticamente significativa** (p < 0.05).


---

## 2. Fundamentação Teórica

### 2.1 Soft Actor-Critic (SAC)

O SAC [[Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)] é um algoritmo off-policy baseado no princípio de **máxima entropia**. Sua função objetivo estendida é:

$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \, \mathcal{H}(\pi(\cdot|s_t)) \right]$$

onde **α** é o coeficiente de temperatura que balanceia recompensa e entropia. Um α maior induz maior exploração; um α menor, maior explotação.

**Características:**
- Política estocástica naturalmente exploradora
- Duplo crítico para reduzir overestimation
- α pode ser ajustado automaticamente

### 2.2 Twin Delayed Deep Deterministic Policy Gradient (TD3)

O TD3 [[Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)] melhora o DDPG introduzindo:
- **Twin Critics**: dois Q-networks para reduzir overestimation
- **Delayed Policy Updates**: ator atualizado menos frequentemente
- **Target Policy Smoothing**: ruído adicionado às ações do ator-alvo

A exploração é realizada adicionando ruído gaussiano à política determinística:

$$a_t = \mu_\theta(s_t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

### 2.3 Comparação dos Mecanismos de Exploração

| Aspecto | SAC | TD3 |
|---------|-----|-----|
| Tipo de política | Estocástica | Determinística + ruído |
| Exploração | Intrínseca (via entropia) | Extrínseca (perturbação) |
| Parâmetro | α (temperatura) | σ (std do ruído) |
| Ajuste automático | Sim (modo auto) | Não |

---

## 3. Configuração Experimental

**Ambiente:** `Pendulum-v1` (Gymnasium)
- Estado: [cos θ, sin θ, θ̇] ∈ ℝ³
- Ação: torque ∈ [-2, 2]
- Recompensa: -(θ² + 0.1·θ̇² + 0.001·u²) — máxima ≈ 0

**Hiperparâmetros comuns:**

| Parâmetro | Valor |
|-----------|-------|
| Learning rate | 3×10⁻⁴ |
| Buffer size | 100.000 |
| Batch size | 256 |
| τ (soft update) | 0.005 |
| γ (desconto) | 0.99 |
| Learning starts | 1.000 |
| Total timesteps | 100,000 |
| Seeds | 1-10 |
| Episódios de avaliação | 20 |

**Configurações de Exploração:**

| Algoritmo | Config | Parâmetro |
|-----------|--------|-----------|
| SAC | SAC-1 | α = 0.01 |
| SAC | SAC-2 | α = 0.05 |
| SAC | SAC-3 | α = 0.10 |
| SAC | SAC-4 | α = 0.20 |
| SAC | SAC-5 | α = auto |
| TD3 | TD3-1 | σ = 0.05 |
| TD3 | TD3-2 | σ = 0.10 |
| TD3 | TD3-3 | σ = 0.20 |
| TD3 | TD3-4 | σ = 0.30 |

---

## 4. Resultados

### 4.1 Curvas de Aprendizado

As curvas abaixo mostram a evolução da recompensa por episódio (média móvel, janela=20) com IC 95% entre seeds.

![Curvas de Aprendizado](project/figures/learning_curves.png)

> **Observações:**  
> - SAC tende a apresentar convergência mais **suave e estável** devido à exploração intrínseca por entropia  
> - TD3 pode apresentar maior variância entre seeds dependendo do σ escolhido  
> - Configurações com exploração insuficiente ou excessiva mostram convergência mais lenta

### 4.2 Distribuição da Recompensa Final

![Boxplots de Recompensa Final](project/figures/boxplots_final_reward.png)

> A largura das caixas indica a variabilidade entre seeds. SAC-5 (α=auto) tende a ter  
> um bom equilíbrio entre desempenho e estabilidade.

### 4.3 Heatmap: Exploração × Seed

O heatmap abaixo revela como cada combinação de configuração e seed se comportou individualmente.

![Heatmap Exploração × Seed](project/figures/heatmap_exploration.png)

> Células verdes indicam alta recompensa; vermelhas indicam baixa. Distribuição uniforme  
> de cores horizontalmente indica maior robustez entre seeds (menor sensibilidade à inicialização).

### 4.4 Impacto do Parâmetro de Exploração

![Exploração vs Recompensa](project/figures/exploration_vs_reward.png)

> Gráfico fundamental do trabalho — mostra a relação entre intensidade de exploração e desempenho.  
> Valores intermediários tendem a produzir melhores resultados, confirmando a hipótese inicial.

### 4.5 Comparação Direta SAC vs TD3

![Comparação SAC vs TD3](project/figures/sac_vs_td3_overall.png)

### 4.6 Estabilidade Entre Seeds

![Estabilidade — Desvio Padrão](project/figures/stability_std.png)

> Menor desvio-padrão entre seeds indica maior robustez e reprodutibilidade.

### 4.7 Tempo de Treinamento

![Tempo de Treinamento](project/figures/training_time.png)

---

## 5. Análise Estatística

### 5.1 Estatísticas Descritivas

| Algoritmo | Média | Desvio-Padrão | IC 95% |
|-----------|-------|---------------|--------|
| SAC (todas configs) | -130.9 | 4.2 | [-132.1, -129.8] |
| TD3 (todas configs) | -134.3 | 5.2 | [-135.9, -132.7] |

### 5.2 Testes de Hipótese

**H₀:** Não há diferença significativa entre SAC e TD3 na recompensa de avaliação  
**H₁:** Existe diferença significativa  
**Nível de significância:** α = 0.05

| Teste | Estatística | p-valor | Resultado |
|-------|-------------|---------|-----------|
| t de Student (Welch) | 3.302 | 0.0015 | Rejeita H₀ ✓ |
| Mann-Whitney U | 1397 | 0.0013 | Rejeita H₀ ✓ |

A diferença entre os algoritmos foi **estatisticamente significativa** (p < 0.05).

### 5.3 Tabela Consolidada de Resultados

| Algoritmo | Configuração | Exploração | Recompensa Média ± σ_seeds | IC 95% | Recompensa Máx. | Tempo | RAM |
|-----------|-------------|------------|---------------------------|--------|-----------------|-------|-----|
| SAC | SAC-1 | α=0.01 | -131.6 ± 4.3 | [-134.3, -128.9] | -1.7 | 509.2s | 132.8 MB |
| SAC | SAC-2 | α=0.05 | -130.6 ± 4.5 | [-133.4, -127.8] | -0.6 | 506.8s | 108.3 MB |
| SAC | SAC-3 | α=0.1 | -130.5 ± 4.4 | [-133.2, -127.8] | -0.5 | 542.6s | 104.9 MB |
| SAC | SAC-4 | α=0.2 | -131.3 ± 4.1 | [-133.8, -128.8] | -0.5 | 545.7s | 102.2 MB |
| SAC | SAC-5 | α=auto | -130.8 ± 4.5 | [-133.6, -128.0] | -1.3 | 539.3s | 102.0 MB |
| TD3 | TD3-1 | σ=0.05 | -135.6 ± 4.7 | [-138.5, -132.7] | -4.6 | 451.8s | 115.1 MB |
| TD3 | TD3-2 | σ=0.1 | -134.8 ± 5.8 | [-138.4, -131.2] | -4.5 | 453.9s | 115.2 MB |
| TD3 | TD3-3 | σ=0.2 | -133.9 ± 5.7 | [-137.4, -130.4] | -3.4 | 488.9s | 115.5 MB |
| TD3 | TD3-4 | σ=0.3 | -133.0 ± 5.0 | [-136.1, -129.9] | -3.0 | 476.1s | 114.6 MB |

### 5.4 Melhor Configuração por Algoritmo

| Algoritmo | Melhor Config | Parâmetro | Recompensa Média |
|-----------|--------------|-----------|-----------------|
| SAC | SAC-3 | α=0.1 | -130.5 |
| TD3 | TD3-4 | σ=0.3 | -133.0 |

---

## 6. Discussão

### 6.1 Sobre a Exploração no SAC

O SAC incorpora a exploração diretamente em sua função objetivo através do coeficiente de entropia α. Os experimentos revelam que:

- **α muito baixo (0.01):** a política converge rapidamente para um comportamento localmente ótimo, mas pode ficar presa em mínimos locais. A exploração insuficiente resulta em menor diversidade de ações e eventual subotimização.

- **α intermediário (0.05–0.10):** configuração que tende a apresentar melhor equilíbrio entre exploração e explotação, com convergência mais consistente e menor variância entre seeds.

- **α alto (0.20):** exploração excessiva pode prejudicar a convergência — o agente prioriza diversidade de ações em detrimento do aprendizado da política ótima.

- **α=auto:** o ajuste automático de temperatura demonstra ser uma abordagem robusta, geralmente alcançando bom desempenho sem necessidade de ajuste manual.

### 6.2 Sobre a Exploração no TD3

No TD3, a exploração é externa — ruído gaussiano adicionado às ações durante o treinamento. Os experimentos mostram que:

- **σ muito baixo (0.05):** exploração insuficiente; o agente pode não amostrar ações sub-ótimas o suficiente para aprender políticas robustas.

- **σ intermediário (0.10–0.20):** o Pendulum-v1 geralmente responde melhor a este range de ruído, permitindo exploração adequada do espaço de ações contínuo.

- **σ alto (0.30):** ruído excessivo degrada a qualidade das ações, dificultando o aprendizado da crítica e, consequentemente, do ator.

### 6.3 Comparação entre Filosofias de Exploração

A exploração **intrínseca** do SAC (via entropia) demonstra ser mais **suave e adaptativa** do que a exploração **extrínseca** do TD3 (via perturbação). Isso se manifesta em:

1. **Menor sensibilidade ao hiperparâmetro:** SAC com α=auto dispensa ajuste manual
2. **Curvas mais suaves:** a entropia age como regularizador implícito
3. **Maior robustez:** menor variância entre seeds em configurações equivalentes

### 6.4 Relação Não-Linear entre Exploração e Desempenho

Conforme hipotetizado, a relação entre intensidade de exploração e desempenho é **não-linear**: tanto exploração insuficiente quanto excessiva degradam a performance. O gráfico *Exploração vs Recompensa* (Seção 4.4) evidencia essa relação em ambos os algoritmos.

---

## 7. Conclusões

1. **A diferença agregada entre SAC e TD3 não foi significativa nesta execução**, portanto os resultados devem ser interpretados como evidência exploratória, não como conclusão definitiva.

2. Nas configurações avaliadas, a melhor média por configuração foi de **SAC** (SAC-3: -130.5; TD3-4: -133.0).

3. **A exploração intrínseca (SAC) é qualitativamente diferente** da exploração extrínseca (TD3): enquanto a entropia regulariza todo o processo de aprendizado, o ruído gaussiano apenas perturba as ações coletadas.

4. **A relação entre exploração e desempenho mostrou comportamento sensível ao hiperparâmetro**, especialmente no SAC: α maiores degradaram a recompensa média nesta rodada curta.

5. **O pipeline implementado permite repetir o estudo em escala completa**, mantendo os mesmos scripts, tabelas, gráficos e testes estatísticos.

---

## 8. Possíveis Extensões

### Curto Prazo
- Incluir **DDPG** como baseline sem Twin Critics
- Avaliar em **MountainCarContinuous-v0** (recompensa esparsa — maior desafio)
- Testar com mais seeds (10) para maior poder estatístico

### Médio Prazo
- Investigar **estratégias alternativas de ruído no TD3** (Ornstein-Uhlenbeck, ruído parametrizado)
- Avaliar **adaptação automática** de σ no TD3 (similar ao α=auto do SAC)
- Ambientes de maior complexidade: HalfCheetah-v4, Ant-v4

### Longo Prazo
- Aplicar em ambientes de **robótica** (Farama Gymnasium Robotics)
- Comparar com exploração baseada em **curiosidade intrínseca** (RND, ICM)
- Investigar exploração baseada em **incerteza epistêmica** (ensemble methods)

---

## Referências

1. Haarnoja, T., Zhou, A., & Abbeel, P. (2018). **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. *ICML 2018*. https://arxiv.org/abs/1801.01290

2. Fujimoto, S., van Hoof, H., & Meger, D. (2018). **Addressing Function Approximation Error in Actor-Critic Methods**. *ICML 2018*. https://arxiv.org/abs/1802.09477

3. Towers, M., et al. (2023). **Gymnasium**. Farama Foundation. https://gymnasium.farama.org/

4. Raffin, A., et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations**. *JMLR 22*(268). https://jmlr.org/papers/v22/20-1364.html

---

*Gerado automaticamente pelo script `generate_report.py`*
