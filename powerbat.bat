@echo off
echo =================================================
echo ATTENTION : L'entrainement YOLO est termine.
echo Le PC va s'eteindre dans 5 minutes (300 sec).
echo Appuyez sur une touche pour ANNULER l'arret.
echo =================================================

:: Lance le compte a rebours de l'arret (300 secondes)
shutdown /s /t 300 /c "Entrainement YOLO fini. Fermeture automatique."

:: Attend une action de l'utilisateur
pause

:: Si l'utilisateur appuie sur une touche, on annule l'arret
shutdown /a
echo Arret annule avec succes !
pause