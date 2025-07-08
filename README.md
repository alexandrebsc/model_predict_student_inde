# Modelo preditivo de INDE do Aluno

Desenvolvido com a proposta de auxiliar no desenvolvimento dos alunos, esse modelo prevê, antes do final do ano letivo, o INDE (Índice de Desenvolvimento Educacional) de um aluno. A previsão é feita com apenas três dos sete indicadores utilizados na ponderação do INDE e algumas informações do aluno. Em mais detalhes, o modelo utiliza:

- a idade do aluno
- a fase atual do aluno
- informação se o aluno é ingressante ou não
- o IAN (Indicador de adequação de nível) do aluno
- o IEG (Indicador de Engajamento) do aluno
- o IAA (Indicador de Autoavaliação) do aluno

Planejado para aplicações reais, note que o IEG e IAA são as únicas informações que precisam ser colhidas. Todas as outras, já estão disponíveis nos dados administrativos do aluno. Mesmo assim, considerando que o IEG já é colhido para o cálculo do INDE, entende-se que essa informação é de fácil acesso. Dessa forma, apenas o valor do IAA do aluno que realmente seria necessário coletar de informações para realizar a previsão. E sendo o IAA um questionário de cinco questões, entende-se que é um esforço aceitável para a previsão do INDE.

Sobre as aplicações práticas, o modelo pode ser disponibilizado para os professores ou para os próprios alunos. Para os docentes é possível a análise de vários alunos de forma eficiente, entendendo quais precisam de apoio. O modelo também pode ser utilizado para que o educador confirme suas intuições sobre alunos individualmente. Já para os alunos, o modelo é ideal para terem visibilidade do seu progresso durante a fase.

Considerando o quão sensível e impactante pode ser para um aluno ter seu INDE previsto, no sentido de se desmotivar por ter um INDE aquém do desejado, ressalta-se que o modelo foi desenvolvido para calcular qual é o previsto se nenhuma ação for tomada. Indicando como o aluno está indo no memomento. Em caso de INDE baixo, o aluno pode experimentar no modelo um IEG maior, pensando "e se eu fizer todas as lições de casas pelo resto do ano". Justamente nesse sentido, ele pode ser utilizado no decorrer do ano, em que o aluno ainda pode mudar o seu IEG e IAA. Contribuindo para o aluno entender a importância do IEG e IAA no impacto indireto em todos outros indicadores. Assim, a previsão não é para limitar as expectativas e sim para auxiliar e motivar o desenvolvimento.

Por fim, seguem algumas informações técnicas sobre o modelo:
- Ele não realiza previsões para alunos da fase 8, por causa das características específicas dessa fase
- Nos dados de testes as estatísticas são:
    - Erro médio absoluto: 0.44 
    - Erro quadrático médio: 0.032
    - Coeficiente de determinação (R²): 0.773
    - Maior erro absoluto: 1.79

