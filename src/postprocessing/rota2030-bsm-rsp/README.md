# Rota2030 Subprojeto 2
## Descrição
Este projeto tem seu foco no entendimento da assinatura de radar gerada pela incidência de ondas eletromagnéticas em uma moto real. Além disso, visa o desenvolvimento de um sistema de alerta ao motorista para o caso de motocicletas que se aproximam pelo corredor.
## Regras
* ## Issues
     Para registrar alterações a serem feitas (melhorias ou correções), criar issues utilizando as tags corretas. É importante que a issue seja fechada quando for resolvida, isso pode ser feito vinculando ela à uma Pull Request, vinculando ela pela mensagem da commit, ou fechando manualmente.

* ## Branching
    Este projeto utilizará o modelo de ramificação [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). O repositório contém uma branch para cada seguimento de iniciação cientifica e projeto de mestrado. Cada uma dessas branchs iniciais será organizada pelo desenvolvedor.

    A branch `main` deverá conter apenas as versões devidamente homologadas e aprovadas pelo pesquisador responsável de cada subprojeto. O versionamento deverá seguir o padrão [Semantic Versioning](https://semver.org/), em que as versões são divididas no formato `MAJOR.MINOR.PATCH`, incrementando conforme:
    - **Major:** Alterações que criem incompatibilidade com versões anteriores (breaking changes)
    - **Minor:** Novas features compatíveis com versões anteriores
    - **Patch:** Correções de bugs compatíveis com versões anteriores

    Para adicionar uma nova feature, uma nova branch deve ser criada e nomeada seguindo o padrão `feature/nome-da-feature`.

    ![gitFlow](gitflow.png)

* ## Commits
    A mensagem das commits deve ser feito em inglês, começando com um verbo no passado, descrevendo sucintamente o que foi realizado. Sempre que existir, vincular a issue na mensagem.

    Exemplos para seguir:

        Fixed AI thread freezing in Samsung devices
        Added a configuration option for switching the model. Closes #10
        Changed the documentation introduction to be more concise
    
    Exemplos para não seguir:

        Fix bugs
        Added #10
        Updated file.md
        Correção no app
        Fixed the AI thread freezing in Samsung devices due to an API implementation bug by their Android changes

