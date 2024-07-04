# speechgen
Speechgen je autoregresivni transformer model treniran da generiše govor srpskog jezika. VQ-VAE komponenta vrši tokenizaciju zvuka, dok je transformer treniran da predviđa sledeći token.
Projekt se takođe sastoji i od transkripcionog modela, čija arhitektura se sastoji od jednog jednodimenzionog konvolucionog sloja, čija je uloga transkripcija teksta.
Model je treniran na maloj količini podataka, koristeći izlaznu reprezentaciju generativnog modela.
# Instrukcije za korišćenje transkripcionog modela

## Dependencies

- Python 3.9
- PyTorch
- FuzzyWuzzy
- Levenshtein

## Težine

- [Link do težina](https://drive.google.com/file/d/1b-8JRvKhzVQ4qoDHm0d0VJi-9-LiO-Y9/view?usp=drive_link)

Težine staviti u folder `transcriber/weights`

## Trening transkripcionog modela

Za pokretanje treninga transkripcionog modela, koristite `train_transcriber.py`.

Na vrhu fajla postoji parametar `dataset_size` koji može biti postavljen na vrednosti 'mini', 'micro' ili 'nano', što određuje veličinu trening skupa.

```bash
python train_transcriber.py
```

## Evaluacija transkripcionog modela
Za evaluaciju transkripcionog modela, koristite `evaluate_transcriber.py`.

```bash
python evaluate_transcriber.py
```