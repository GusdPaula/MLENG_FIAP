# 🔐 Guia de Segurança: Usando DVC, API, MLflow e Grafana Sem Expor Segredos

Este guia explica como usar com segurança todos os componentes da infraestrutura MLOps sem expor informações sensíveis como chaves de API, senhas ou credenciais.

---

## Sumário

1. [Princípios de Segurança](#princípios-de-segurança)
2. [Configuração Segura do DVC](#configuração-segura-do-dvc)
3. [Deploy Seguro da API](#deploy-seguro-da-api)
4. [Acesso Seguro ao MLflow](#acesso-seguro-ao-mlflow)
5. [Configuração Segura do Grafana](#configuração-segura-do-grafana)
6. [Segredos no GitHub Actions](#segredos-no-github-actions)
7. [Segurança no Desenvolvimento Local](#segurança-no-desenvolvimento-local)
8. [Rotação de Segredos](#rotação-de-segredos)
9. [Solução de Problemas](#solução-de-problemas)

---

## Princípios de Segurança

### Regras Básicas de Segurança

- **Nunca faça commit de segredos no controle de versão** - Use `.gitignore` e variáveis de ambiente
- **Use AWS Secrets Manager/Parameter Store** para segredos em produção
- **Aplique o princípio de menor privilégio** - Conceda apenas as permissões mínimas necessárias
- **Roteie credenciais regularmente** - Implemente rotação automatizada quando possível
- **Audite o acesso** - Monitore quem acessa quais recursos
- **Criptografe tudo** - Use criptografia em repouso e em trânsito

### Arquivos Que Nunca Devem Conter Segredos

- Arquivos `.env` (use `.env.example` como modelo)
- Arquivos Terraform `.tfvars` (use passagem segura de variáveis)
- Arquivos de configuração com credenciais hardcoded
- Scripts com senhas embutidas
- Documentação com credenciais reais

---

## Configuração Segura do DVC

### Visão Geral

O DVC usa AWS S3 como armazenamento remoto com usuários IAM para autenticação. As credenciais são gerenciadas via chaves de acesso AWS IAM, não hardcoded em arquivos de configuração.

### Configuração Segura do DVC

#### 1. Configure o DVC Remote (Sem Segredos no Config)

O arquivo `.dvc/config` contém apenas a URL do bucket, sem credenciais:

```ini
[core]
    analytics = false
    remote = s3-infra
    autostage = true

['remote "s3-infra"']
    url = s3://fiap-ml-dvc-bucket-tech-challenger-3z9kyqs3
```

**✅ SEGURO**: Nenhuma credencial armazenada no arquivo de configuração.

#### 2. Configure Credenciais AWS com Segurança

**Opção A: Perfil AWS CLI (Recomendado para Desenvolvimento)**

```bash
# Configure perfil AWS com credenciais de usuário IAM
aws configure --profile dvc-profile

# Insira as credenciais quando solicitado (nunca armazenadas em arquivos)
# AWS Access Key ID: <sua-access-key>
# AWS Secret Access Key: <sua-secret-key>
# Default region: us-east-1
# Output format: json
```

Configure o DVC para usar o perfil:

```bash
dvc remote modify s3-infra profile dvc-profile
```

**Opção B: Variáveis de Ambiente (Recomendado para CI/CD)**

```bash
export AWS_ACCESS_KEY_ID=<sua-access-key>
export AWS_SECRET_ACCESS_KEY=<sua-secret-key>
export AWS_DEFAULT_REGION=us-east-1
```

**Opção C: Função IAM AWS (Recomendado para EC2/ECS)**

Para recursos rodando na AWS (EC2, ECS), use funções IAM em vez de credenciais:

```hcl
# Exemplo: Função de tarefa ECS com acesso S3
resource "aws_iam_role_policy" "ecs_task_s3_policy" {
  name = "${var.project_name}-ecs-task-s3-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::fiap-ml-dvc-bucket-tech-challenger-3z9kyqs3",
          "arn:aws:s3:::fiap-ml-dvc-bucket-tech-challenger-3z9kyqs3/*"
        ]
      }
    ]
  })
}
```

#### 3. Usando Comandos DVC com Segurança

```bash
# Pull de dados (usa credenciais configuradas)
dvc pull

# Push de dados (usa credenciais configuradas)
dvc push

# Nenhuma credencial necessária no comando
dvc repro
```

### Gerenciamento de Usuários IAM

A infraestrutura cria dois usuários IAM para acesso DVC:

#### Usuário Leitura/Escrita (para desenvolvedores)

```bash
# Obter nome do usuário via output do Terraform
cd infra-api/s3
terraform output -raw dvc_user_name

# Criar chaves de acesso via Console AWS (nunca via CLI por segurança)
# 1. Vá em IAM → Users → <dvc_user_name>
# 2. Credenciais de segurança → Create access key
# 3. Escolha "Application running outside AWS"
# 4. Salve credenciais com segurança (gerenciador de senhas)
```

#### Usuário Apenas Leitura (para membros da equipe que só precisam acessar dados)

```bash
# Obter nome do usuário read-only
terraform output -raw dvc_readonly_user_name

# Criar chaves de acesso via Console AWS
# Use o mesmo processo acima
```

### Melhores Práticas de Segurança para DVC

- **Nunca faça commit** de arquivos `.aws/credentials` ou `.aws/config`
- **Use perfis diferentes** para diferentes projetos/ambientes
- **Roteie chaves de acesso** a cada 90 dias
- **Desative chaves não utilizadas** imediatamente
- **Use MFA** para acesso ao console AWS
- **Monitore CloudTrail** para logs de acesso S3

---

## Deploy Seguro da API

### Visão Geral

A API usa ECS Fargate com valores sensíveis passados como variáveis de ambiente. A chave da API é passada via variáveis Terraform, não hardcoded.

### Deploy Seguro da API

#### 1. Gere Chave de API com Segurança

```bash
# Gere uma chave de API forte (nunca use senhas fracas)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Armazene a chave gerada em um gerenciador de senhas ou AWS Secrets Manager.

#### 2. Deploy da API com Variáveis Seguras

**Opção A: Passagem de Variáveis via Linha de Comando (Nunca faça commit disso)**

```bash
cd infra-api/api

# Passe a chave da API via linha de comando (não armazenada em arquivos)
terraform apply \
  -var="api_key=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")" \
  -var="mlflow_tracking_uri=https://seu-mlflow-url.cloudfront.net"
```

**Opção B: Variáveis de Ambiente (Recomendado para CI/CD)**

```bash
# Configure variável de ambiente
export TF_VAR_api_key=<sua-api-key>

# Terraform pega automaticamente variáveis TF_VAR_*
terraform apply
```

**Opção C: Arquivo .tfvars Seguro (Nunca faça commit no Git)**

Crie `terraform.tfvars` (adicione ao `.gitignore`):

```hcl
api_key              = "<sua-api-key>"
mlflow_tracking_uri  = "https://seu-mlflow-url.cloudfront.net"
```

Adicione ao `.gitignore`:

```gitignore
# Arquivos de variáveis Terraform com segredos
*.tfvars
!example.tfvars
```

**Opção D: AWS Secrets Manager (Recomendado para Produção)**

```hcl
# Armazene chave da API no Secrets Manager
resource "aws_secretsmanager_secret" "api_key" {
  name = "mlflow-fiap/api-key"
}

resource "aws_secretsmanager_secret_version" "api_key" {
  secret_id     = aws_secretsmanager_secret.api_key.id
  secret_string = var.api_key
}

# Referencie na definição de tarefa ECS
secrets = [
  {
    name      = "API_KEY"
    value_from = aws_secretsmanager_secret_version.api_key.arn
  }
]
```

#### 3. Verifique que a Chave da API Não Está Exposta

```bash
# Verifique estado do Terraform (deve mostrar dados sensíveis como <sensitive>)
terraform output api_key

# Verifique variáveis de ambiente na tarefa em execução
aws ecs describe-tasks \
  --cluster mlflow-fiap-api-cluster \
  --tasks <task-id> \
  --query 'tasks[0].containers[0].environment'
```

### Recursos de Segurança da API

A infraestrutura da API inclui várias medidas de segurança:

#### 1. Autenticação com Chave de API

```python
# API valida a chave em cada requisição
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Chave de API inválida")
    return x_api_key
```

#### 2. Limitação de Taxa (Rate Limiting)

```python
# Configure limitação de taxa no .env
RATE_LIMIT=100/minute
```

#### 3. Whitelist de IPs (Opcional)

```python
# Configure IPs permitidos no .env
ALLOWED_IPS=192.168.1.100,10.0.0.50
```

#### 4. Apenas HTTPS

CloudFront impõe HTTPS:

```hcl
viewer_protocol_policy = "redirect-to-https"
```

### Melhores Práticas de Segurança para API

- **Roteie chaves de API** regularmente (a cada 30-90 dias)
- **Use chaves diferentes** para diferentes ambientes
- **Monitore uso da API** via métricas CloudWatch
- **Configure alertas** para atividade suspeita
- **Use chaves de curta duração** quando possível
- **Implemente restrições CORS** se necessário

---

## Acesso Seguro ao MLflow

### Visão Geral

O MLflow usa RDS PostgreSQL com senha armazenada no AWS Secrets Manager. A instância EC2 recupera a senha em tempo de execução via função IAM.

### Configuração Segura do MLflow

#### 1. Gerenciamento de Senha do Banco de Dados

A infraestrutura automaticamente:

```hcl
# Gera senha aleatória
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Armazena no Secrets Manager
resource "aws_secretsmanager_secret" "db_password_secret" {
  name = "mlflow-fiap/db-password"
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password_secret.id
  secret_string = random_password.db_password.result
}
```

**✅ SEGURO**: A senha nunca aparece no estado ou logs do Terraform.

#### 2. EC2 Recupera Senha em Tempo de Execução

O script user_data recupera a senha:

```bash
#!/bin/bash
# Obter senha do DB do Secrets Manager
DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id ${aws_secretsmanager_secret.db_password_secret.id} \
  --query SecretString --output text)

# Iniciar MLflow com backend PostgreSQL
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow_user:${DB_PASSWORD}@${aws_db_instance.mlflow_db.endpoint}:5432/mlflow \
  --default-artifact-root "s3://${aws_s3_bucket.mlflow_artifacts.bucket}" \
  --serve-artifacts
```

**✅ SEGURO**: A senha existe apenas na memória, nunca no disco.

#### 3. Função IAM para Acesso a Segredos

```hcl
resource "aws_iam_role_policy" "mlflow_ec2_secrets_policy" {
  name = "${var.project_name}-mlflow-ec2-secrets-policy"
  role = aws_iam_role.mlflow_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "secretsmanager:GetSecretValue"
        Resource = aws_secretsmanager_secret.db_password_secret.arn
      }
    ]
  })
}
```

**✅ SEGURO**: Apenas a instância EC2 do MLflow pode acessar o segredo.

#### 4. Acessando a UI do MLflow

O MLflow é acessado via CloudFront (apenas HTTPS):

```bash
# URL de produção
https://mlflow.asgardprint.com.br

# Ou URL do CloudFront
https://dxxxxxxxx.cloudfront.net
```

**Recursos de Segurança:**
- HTTPS imposto pelo CloudFront
- Sem acesso direto à instância EC2
- Grupo de segurança permite apenas tráfego do CloudFront
- RDS em subnet privada, sem acesso público

### Autenticação do MLflow (Opcional)

Para segurança adicional, habilite autenticação do MLflow:

#### Opção A: Autenticação Básica

```bash
# Configure variáveis de ambiente
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=<senha-segura>

# Inicie MLflow com auth
mlflow server \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://... \
  --app-name basic-auth
```

#### Opção B: Integração OAuth

Configure provedor OAuth (GitHub, Google, etc.):

```python
# mlflow/server/auth.py
from mlflow.server.auth import auth

# Configure OAuth
auth_config = {
    "provider": "github",
    "client_id": os.getenv("GITHUB_CLIENT_ID"),
    "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
}
```

### Melhores Práticas de Segurança para MLflow

- **Use PostgreSQL** em vez de SQLite para produção
- **Habilite backups RDS** (já configurado: retenção de 7 dias)
- **Criptografe armazenamento RDS** (já habilitado)
- **Restrinja security groups** apenas para CloudFront
- **Monitore acesso a experimentos** via CloudTrail
- **Roteie regularmente** senhas do banco de dados
- **Use subnets privadas** para RDS (já configurado)

---

## Configuração Segura do Grafana

### Visão Geral

O Grafana usa uma senha de administrador gerada aleatoriamente que é passada para a instância EC2 via user_data. A senha é gerada no momento do deploy e nunca armazenada em arquivos.

### Configuração Segura do Grafana

#### 1. Geração de Senha

A infraestrutura gera uma senha aleatória:

```hcl
resource "random_password" "grafana_admin" {
  length  = 32
  special = false
}
```

#### 2. Senha Passada via User Data

```hcl
user_data = templatefile("${path.module}/user_data.sh", {
  prometheus_url = var.prometheus_url
  grafana_admin_password = random_password.grafana_admin.result
})
```

#### 3. Configuração do Grafana

O script user_data configura o Grafana:

```bash
# Configure Grafana com senha gerada
cat > /etc/grafana/grafana.ini << EOF
[security]
admin_user = admin
admin_password = ${grafana_admin_password}
allow_embedding = true
cookie_secure = true
cookie_samesite = lax
content_security_policy = true
strict_transport_security = true
EOF
```

**✅ SEGURO**: A senha existe apenas na configuração do Grafana, não no código fonte.

#### 4. Acessando o Grafana

O Grafana é acessado via CloudFront:

```bash
# URL de produção
https://d3naqrkpy0vqtm.cloudfront.net

# URL do Dashboard
https://d3naqrkpy0vqtm.cloudfront.net/d/a4vkb7/api-overview
```

#### 5. Recuperando Senha de Administrador

Após o deploy, recupere a senha:

```bash
# Opção 1: Do output do Terraform
cd infra-api/grafana-ec2
terraform output -raw grafana_admin_password

# Opção 2: Da instância EC2 via SSM
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /etc/grafana/grafana.ini | grep admin_password"]'

# Opção 3: SSH na instância (se o grupo de segurança permitir)
ssh -i <key-pair> ubuntu@<public-ip>
sudo cat /etc/grafana/grafana.ini | grep admin_password
```

### Recursos de Segurança do Grafana

#### 1. Acesso Anônimo (Função de Visualizador)

```ini
[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer
```

**Implicação de Segurança**: Usuários podem visualizar dashboards sem autenticação mas não podem modificar nada.

#### 2. Configuração SSL/TLS

```ini
[security]
cookie_secure = true
strict_transport_security = true
```

#### 3. Política de Segurança de Conteúdo

```ini
[security]
content_security_policy = true
x_content_type_options = true
x_xss_protection = true
```

#### 4. Proxy Reverso Nginx

O Nginx fornece camada de segurança adicional:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/grafana.crt;
    ssl_certificate_key /etc/nginx/ssl/grafana.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

### Melhores Práticas de Segurança para Grafana

- **Altere a senha de administrador** após o primeiro login
- **Crie usuários adicionais** com permissões apropriadas
- **Desabilite acesso anônimo** se não for necessário
- **Habilite log de auditoria** para ambientes sensíveis
- **Use apenas HTTPS** (já imposto via CloudFront)
- **Atualize regularmente** o Grafana para a versão mais recente
- **Configure backup** para dashboards e configurações do Grafana

---

## Segredos no GitHub Actions

### Visão Geral

O GitHub Actions usa OIDC (OpenID Connect) para acesso seguro à AWS sem armazenar credenciais de longa duração.

### Configuração Segura do GitHub Actions

#### 1. Configure Provedor OIDC

```hcl
# Crie provedor OIDC para GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com"
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831b3d0c6975"
  ]
}
```

#### 2. Crie Função IAM para GitHub Actions

```hcl
resource "aws_iam_role" "github_actions" {
  name = "github-actions-mlflow-fiap"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:GusdPaula/MLENG_FIAP:*"
        }
      }
    }]
  })
}
```

#### 3. Anexe Política à Função

```hcl
resource "aws_iam_role_policy_attachment" "github_actions_attach" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.github_actions_policy.arn
}
```

#### 4. Configure Workflow do GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy Infrastructure

on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<account-id>:role/github-actions-mlflow-fiap
          aws-region: us-east-1

      - name: Deploy with Terraform
        run: |
          cd infra-api/api
          terraform init
          terraform apply -auto-approve \
            -var="api_key=${{ secrets.API_KEY }}"
```

#### 5. Armazene Segredos no GitHub

Vá em Settings → Secrets and variables → Actions do repositório:

- **API_KEY**: Chave de API gerada para a API
- **DOCKERHUB_USERNAME**: Nome de usuário do Docker Hub para push de imagens
- **DOCKERHUB_TOKEN**: Token de acesso do Docker Hub

**✅ SEGURO**: Os segredos são criptografados e nunca expostos em logs.

### Melhores Práticas de Segurança para GitHub Actions

- **Use OIDC** em vez de chaves de acesso de longa duração
- **Restrinja acesso ao repositório** nas condições da função IAM
- **Use segredos de ambiente** para diferentes ambientes
- **Roteie segredos** regularmente
- **Monitore execuções de workflow** para atividade suspeita
- **Use revisores obrigatórios** para workflows sensíveis
- **Fixe versões de actions** para prevenir ataques de cadeia de suprimentos

---

## Segurança no Desenvolvimento Local

### Variáveis de Ambiente

Use o arquivo `.env` para desenvolvimento local (nunca faça commit):

```bash
# Copie arquivo de exemplo
cp .env.example .env

# Edite com seus valores
nano .env
```

**`.env.example` (seguro para commit):**

```dotenv
# Configuração MLflow
MLFLOW_TRACKING_URI=https://mlflow.asgardprint.com.br

# Configuração AWS
AWS_DEFAULT_REGION=us-east-1
AWS_REGION=us-east-1
AWS_PROFILE=aws

# Configuração API
API_KEY=

# Configuração de Segurança
RATE_LIMIT=100/minute
ALLOWED_IPS=

# Configuração de Modelo
MLFLOW_MODEL_ALIAS=champion
```

**`.env` (nunca faça commit):**

```dotenv
# Configuração MLflow
MLFLOW_TRACKING_URI=https://mlflow.asgardprint.com.br

# Configuração AWS
AWS_DEFAULT_REGION=us-east-1
AWS_REGION=us-east-1
AWS_PROFILE=aws

# Configuração API
API_KEY=sua-chave-api-gerada-aqui

# Configuração de Segurança
RATE_LIMIT=100/minute
ALLOWED_IPS=

# Configuração de Modelo
MLFLOW_MODEL_ALIAS=champion
```

### Configuração Git

Certifique-se de que `.gitignore` inclui:

```gitignore
# Variáveis de ambiente
.env
.env.local
.env.*.local

# Credenciais AWS
.aws/credentials
.aws/config

# Terraform
*.tfvars
!example.tfvars
.terraform/
*.tfstate
*.tfstate.*

# Python
__pycache__/
*.pyc
.venv/
```

### Gerenciamento de Segredos Locais

#### Opção A: Python-dotenv

```python
# Carregue variáveis de ambiente do .env
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
```

#### Opção B: Keyring (Multiplataforma)

```python
import keyring

# Armazene segredo
keyring.set_password("mlflow-fiap", "api_key", "sua-chave-api")

# Recupere segredo
api_key = keyring.get_password("mlflow-fiap", "api_key")
```

#### Opção C: AWS Secrets Manager (para dev local)

```python
import boto3
import json

client = boto3.client('secretsmanager', region_name='us-east-1')

response = client.get_secret_value(SecretId='mlflow-fiap/api-key')
secret = json.loads(response['SecretString'])
api_key = secret['api_key']
```

### Melhores Práticas de Segurança para Desenvolvimento Local

- **Nunca faça commit** de arquivos `.env`
- **Use credenciais diferentes** para desenvolvimento e produção
- **Roteie credenciais locais** regularmente
- **Use VPN** ao acessar recursos de produção
- **Mantenha software atualizado** (SO, Python, dependências)
- **Use firewall** para restringir conexões de entrada
- **Habilite criptografia de disco** em máquinas de desenvolvimento

---

## Rotação de Segredos

### Estratégias de Rotação Automatizada

#### 1. Rotação de Chave de API

```bash
# Gere nova chave
NEW_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Atualize infraestrutura
cd infra-api/api
terraform apply -var="api_key=$NEW_KEY"

# Atualize Segredos do GitHub
# Vá em Settings → Secrets → Actions → API_KEY
# Cole nova chave

# Atualize .env local
sed -i "s/API_KEY=.*/API_KEY=$NEW_KEY/" .env
```

#### 2. Rotação de Chave de Acesso AWS

```bash
# Crie nova chave de acesso
aws iam create-access-key --user-name <user-name>

# Atualize credenciais AWS locais
aws configure --profile dvc-profile

# Teste novas credenciais
aws s3 ls s3://fiap-ml-dvc-bucket-tech-challenger-3z9kyqs3 --profile dvc-profile

# Exclua chave antiga
aws iam delete-access-key --user-name <user-name> --access-key-id <old-key-id>
```

#### 3. Rotação de Senha do Banco de Dados

```bash
# Gere nova senha
NEW_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Atualize senha do RDS
aws rds modify-db-instance \
  --db-instance-identifier mlflow-fiap-db \
  --master-user-password "$NEW_PASSWORD" \
  --apply-immediately

# Atualize Secrets Manager
aws secretsmanager put-secret-value \
  --secret-id mlflow-fiap/db-password \
  --secret-string "$NEW_PASSWORD"

# Reinicie servidor MLflow para pegar nova senha
# (via script user_data ou comando SSM)
```

#### 4. Rotação de Senha de Administrador do Grafana

```bash
# Gere nova senha
NEW_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Atualize via API do Grafana
curl -X PUT \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"password":"'"$NEW_PASSWORD"'"}' \
  http://localhost:3000/api/admin/users/1/password

# Ou atualize via user_data e redeploy
cd infra-api/grafana-ec2
terraform apply
```

### Cronograma de Rotação

| Tipo de Segredo | Frequência de Rotação | Método |
|-----------------|----------------------|--------|
| Chaves de API | A cada 30-90 dias | Manual/automatizado |
| Chaves de Acesso AWS | A cada 90 dias | Manual |
| Senhas do Banco de Dados | A cada 90 dias | Automatizado via Lambda |
| Senha do Grafana | A cada 180 dias | Manual |
| Chaves SSH | A cada 180 dias | Manual |

### Rotação Automatizada com Lambda

Crie função Lambda para rotacionar senha do RDS:

```python
import boto3
import json
import secrets

def lambda_handler(event, context):
    client = boto3.client('rds')
    secrets = boto3.client('secretsmanager')

    # Gere nova senha
    new_password = secrets.token_urlsafe(32)

    # Atualize RDS
    client.modify_db_instance(
        DBInstanceIdentifier='mlflow-fiap-db',
        MasterUserPassword=new_password,
        ApplyImmediately=True
    )

    # Atualize Secrets Manager
    secrets.put_secret_value(
        SecretId='mlflow-fiap/db-password',
        SecretString=new_password
    )

    return {'statusCode': 200, 'body': json.dumps('Senha rotacionada')}
```

Agende com CloudWatch Events (expressão cron):

```json
{
  "scheduleExpression": "rate(90 days)"
}
```

---

## Solução de Problemas

### Problemas Comuns de Segurança

#### 1. Erros "Acesso Negado"

**Problema**: DVC ou API retorna acesso negado.

**Soluções**:
```bash
# Verifique credenciais AWS
aws sts get-caller-identity --profile dvc-profile

# Verifique permissões IAM
aws iam list-user-policies --user-name <user-name>

# Verifique política do bucket S3
aws s3api get-bucket-policy --bucket fiap-ml-dvc-bucket-tech-challenger-3z9kyqs3
```

#### 2. Erros "Invalid API Key"

**Problema**: API retorna 403 Proibido.

**Soluções**:
```bash
# Verifique chave de API no ambiente
echo $API_KEY

# Verifique ambiente da tarefa ECS
aws ecs describe-tasks \
  --cluster mlflow-fiap-api-cluster \
  --tasks <task-id> \
  --query 'tasks[0].containers[0].environment'

# Regere chave de API se necessário
```

#### 3. "Connection Refused" ao MLflow

**Problema**: Não é possível conectar ao servidor MLflow.

**Soluções**:
```bash
# Verifique status da instância EC2
aws ec2 describe-instances --instance-ids <instance-id>

# Verifique regras do security group
aws ec2 describe-security-groups --group-ids <sg-id>

# Verifique distribuição CloudFront
aws cloudfront get-distribution --id <distribution-id>
```

#### 4. Estado do Terraform Contém Segredos

**Problema**: Dados sensíveis no estado do Terraform.

**Soluções**:
```bash
# Remova recurso sensível do estado
terraform state rm <resource-name>

# Recrie com gerenciamento adequado de segredos
terraform apply

# Habilite criptografia de estado (se ainda não)
# Atualize configuração do backend:
# encrypt = true
```

### Comandos de Auditoria de Segurança

#### Verifique Segredos Expostos

```bash
# Busque segredos potenciais no histórico Git
git log -p --all -S 'password' -S 'api_key' -S 'secret'

# Use truffleHog para escanear repositório
trufflehog --regex --entropy=False /path/to/repo

# Use git-secrets para prevenir commits futuros
git secrets --register-aws
git secrets --scan
```

#### Verifique Permissões IAM

```bash
# Simule política IAM
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::<account-id>:user/<user-name> \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::bucket-name/*

# Liste chaves de acesso
aws iam list-access-keys --user-name <user-name>

# Verifique último uso da chave
aws iam get-access-key-last-used \
  --access-key-id <key-id>
```

#### Monitore CloudTrail

```bash
# Busque chamadas de API suspeitas
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::S3::Bucket \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ)

# Verifique tentativas de autenticação falhadas
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=ConsoleLogin
```

### Lista de Verificação de Segurança

Antes de fazer deploy para produção:

- [ ] Sem segredos em arquivos `.env` (use `.env.example`)
- [ ] Sem segredos em arquivos Terraform (use variáveis)
- [ ] Sem segredos no histórico Git
- [ ] Chaves de API passadas via variáveis de ambiente ou Secrets Manager
- [ ] Senhas do banco de dados no Secrets Manager
- [ ] Funções IAM usadas em vez de chaves de acesso quando possível
- [ ] Grupos de segurança seguem princípio de menor privilégio
- [ ] Buckets S3 têm criptografia habilitada
- [ ] Buckets S3 têm versionamento habilitado
- [ ] Instâncias RDS têm criptografia habilitada
- [ ] Instâncias RDS têm backups habilitados
- [ ] CloudFront impõe HTTPS
- [ ] Estado do Terraform é criptografado
- [ ] GitHub Actions usa OIDC
- [ ] Todas as credenciais rotacionadas nos últimos 90 dias
- [ ] MFA habilitado para acesso ao console AWS
- [ ] CloudTrail habilitado para auditoria
- [ ] Alarmes CloudWatch configurados para eventos de segurança

---

## Recursos Adicionais

### Documentação de Segurança AWS

- [Melhores Práticas de Segurança da AWS](https://docs.aws.amazon.com/whitepapers/latest/aws-security-best-practices/)
- [Melhores Práticas do IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Documentação do Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/)
- [Documentação do CloudTrail](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/)

### Segurança Terraform

- [Melhores Práticas de Segurança do Terraform](https://www.terraform.io/docs/cloud/guides/recommended-practices/security.html)
- [Segurança do Estado do Terraform](https://www.terraform.io/docs/cloud/state/index.html#state-encryption)

### Ferramentas de Segurança

- **tfsec**: Scanner de segurança Terraform
- **checkov**: Scanner de segurança de infraestrutura como código
- **truffleHog**: Scanner de segredos para repositórios Git
- **Prowler**: Ferramenta de auditoria de segurança AWS
- **AWS Security Hub**: Monitoramento de segurança centralizado

### Monitoramento e Alertas

- Configure alarmes CloudWatch para:
  - Tentativas de autenticação de API falhadas
  - Padrões de acesso S3 incomuns
  - Mudanças de política IAM
  - Mudanças de regras de security group
  - Falhas de login no console

---

## Conclusão

Esta infraestrutura implementa múltiplas camadas de segurança para proteger informações sensíveis:

1. **Sem segredos hardcoded** em arquivos de configuração
2. **AWS Secrets Manager** para senhas do banco de dados
3. **Funções IAM** em vez de credenciais de longa duração
4. **Variáveis de ambiente** para segredos em tempo de execução
5. **GitHub Actions OIDC** para autenticação CI/CD
6. **Criptografia em repouso** para S3 e RDS
7. **Apenas HTTPS** para todo acesso externo
8. **Grupos de segurança** seguindo princípio de menor privilégio
9. **Rotação regular** de credenciais

Seguindo este guia, você pode usar DVC, API, MLflow e Grafana com segurança sem expor informações sensíveis. Auditorias de segurança regulares e rotação de credenciais são essenciais para manter uma postura de segurança forte.
