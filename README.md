# speechgen
Speechgen je autoregresivni transformer model treniran da generiše govor srpskog jezika. VQ-VAE komponenta vrši tokenizaciju zvuka,
dok je transformer treniran da predviđa sledeći token.
Projekat se takođe sadrži i transkripcioni model, čija arhitektura se sastoji iz jednog jednodimenzionog konvolucionog sloja, čija je uloga transkripcija teksta.
Ovaj transkripcioni model je treniran na maloj količini podataka, koristeći izlaznu reprezentaciju generativnog modela.
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

## Primeri generisanih zvukova

Sledeći zvukovi su dužine 12 sekundi, gde je prvih 8 sekundi ulazni kontekst, dok su naredne 4 sekunde izgenerisane pomoću generatora.
Uzeti u obzir da osim konteksta nema nijednog drugog ulaza u transformer - nije mu rečeno šta da kaže, već on slobodno generiše
nastavak govora. Model takođe nije treniran na tekstu, tako da je impresivno što je uspeo da nauči neke česte reči samo slušanjem.

<video src='generated_examples/1.mp4' width=180 />
<video src='generated_examples/2.mp4' width=180 />
