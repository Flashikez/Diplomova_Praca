Obsahuje experimenty trénovanie a testovania GANov s percepčnou chybovou funkciou generátora.
Každý priečinok obsahuje experiment podľa názvu priečinku.

- Každý experiment je v samostatnom priečinku v ktorom sa nachádza:
	-pretrained_models - predtrénované modely generátora a diskriminátora použité v experimente
	- train_súbor - skript na spustenie tréningu GANu
	-plot - vývoj zväčšovaných obrázkov počas tréningu
	-generator_test - skript, ktorý otestuje natrénovaný generátor, výstup bude uložený do priečinku log
	-log - výsledné hodnoty metrík natrénovaného generátora, prípadne ostatných zväčšovacích metód
	-generates_samples - nachádza sa len v niektorých experimentoch. Vytvorí obrázok príkladov zväčšenia generátorom alebo inou metódou.