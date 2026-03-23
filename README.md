# Desafio MBA Engenharia de Software com IA - Full Cycle

1. **Criar e ativar um ambiente virtual (`venv`):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

2. **Instalar as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar as variáveis de ambiente:**

   - Duplique o arquivo `.env.example` e renomeie para `.env`
   - Abra o arquivo `.env` e substitua os valores pelas suas chaves de API reais obtidas conforme instruções abaixo
   - OBS: Por enquanto só existe implementação para OPENAI API, para GOOGLE_API_KEY ainda não foi implementado

4. **Subir o banco de dados:**

   ```bash
   docker compose up -d
   ```

5. **Ingerir o PDF no banco vetorial:**

   ```bash
   python src/ingest.py
   ```

6. **Execução**

   ```bash
   python src/chat.py
   ```