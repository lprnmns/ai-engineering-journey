# CLI Cheatsheet

## ls

Dosya ve klasörleri listeler.

```bash
ls -la
```

## pwd

Şu an bulunduğun klasörün tam yolunu gösterir.

```bash
pwd
```

## grep

Metin içinde arama yapar.

```bash
grep -n "TODO" -R .
```

## find

Dosya veya klasör arar.

```bash
find . -name "*.py"
```

## cat

Dosya içeriğini terminale basar.

```bash
cat README.md
```

## pipe |

Bir komutun çıktısını başka komuta aktarır.

```bash
grep "ERROR" app.log
```

Not: Aşağıdaki kullanım da çalışır ama çoğu durumda gereksizdir:

```bash
cat app.log | grep "ERROR"
```
