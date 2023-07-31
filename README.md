# Multipanel Sound Transmission Loss Lumped Parameter Model
Python code for generating large datasets for sound transmission loss in multi-panel (single, double, or triple panels) with and without absorbers
use file stl_evaluation_multipanel.py
(or notebook file .ipynb. Note that only .py file will be updated in future, not the Jupyter notebook)

Example materials for panels and absorbers are in Materials folder. If you wish to provide more data, you can add that to those files. The data have been accumulated from various sources over a period of time.

The equations for the model are taken from Marshall Long's Architectural Acoustics book. For absorber transmission loss, the Miki Model is used. For double and triple panel with absorbers in-between, Fahy's 1987 formula is employed. Details inside the code.

OITC (Outside Inside Transmission Class) rating included to calculate over user provided range vs. 50-1000 Hz as listed in ASTM E1332-10a method. 

## Bibliography:
- Long M. Architectural acoustics. Elsevier; 2005 Dec 23.
- ASTM International. ASTM E1332-10a, standard classification for rating outdoor-indoor sound attenuation. 2010.
- Matyka M, Koza Z. How to calculate tortuosity easily?. InAIP Conference Proceedings 4 2012 May 15 (Vol. 1453, No. 1, pp. 17-22). American Institute of Physics.
- Miki Y. Acoustical properties of porous materials-generalizations of empirical models. Journal of the Acoustical Society of Japan (E). 1990;11(1):25-8.
- Fahy F. 4. In: Sound and structural vibration: Radiation, transmission and response. First ed. London: Acad. Press; 1987. p. 143–215. 
- Drozdov AD, de Claville Christiansen J. The effect of porosity on elastic moduli of polymer foams. Journal of Applied Polymer Science. 2020 Mar 10;137(10):48449.
- Vinokur RY. Transmission loss of triple partitions at low frequencies. Applied Acoustics. 1990 Jan 1;29(1):15-24.
- Xin FX, Lu TJ. Analytical modeling of sound transmission through clamped triple-panel partition separated by enclosed air cavities. European Journal of Mechanics-A/Solids. 2011 Nov 1;30(6):770-82.
