Experiment dvojnásobného zväčšenia rozlíšenia rozšíreného datasetu Text_Set
V tomto prípade boli uložené všetky generátory a diskriminátory, teda po každej epoche.
Následne sa vybrali tri najlepšie a tie sa testovali. (test_top3_epochs)
Obsahuje: 
-examples - vygenerované príklady zväčšenia generátorom/inými metódami pomocou skriptu generate_samples alebo compare
-...train - spustí tréning GANu-test_generator_on_tesseract

--POZNAMKA
Keďže diskriminátory zaberajú najviac miesta na disku a po vykonaní tréningu ich už nyvužívame
v tomto prípade ich z kapacitných dôvodov v prílohách neuvádzame.