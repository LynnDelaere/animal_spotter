# Self-hosted GitHub Actions runner (Docker)

Deze map bevat de configuratie om een self-hosted GitHub Actions runner
voor deze repository in een Docker-container te draaien.

## Gebruik (op een VM met Docker)

1. Kopieer het voorbeeldbestand:

```bash
cp docker-compose.runner.example.yml docker-compose.runner.yml
```

2. Vraag een token aan voor de runner:
    - Ga naar de repository op GitHub
    - Ga naar "Settings" > "Actions" > "Runners"
    - Klik op "New self-hosted runner"
    - Volg de stappen tot je bij "Get token" komt en kopieer het token 

3. Plak het token in `docker-compose.runner.yml` bij `RUNNER_TOKEN`

4. Start de runner:
```bash
docker compose -f docker-compose.runner.yml up -d
```

5. Controleer of de runner actief is op GitHub bij "Settings" > "Actions" > "Runners"