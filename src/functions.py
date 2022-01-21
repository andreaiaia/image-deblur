import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy.optimize import minimize
from numpy import fft

# Creazione Kernel Gaussiano di dimensione kerneln e deviazione standard sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)
    # Kernel Gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel Gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()


# Calcolo FFT del kernel 'K' di dimensioni 'd' con opportuno padding necessario a far combaciare la dimensione di 'shape'

def psf_fft(K, d, shape):
    '''
    @param K: the gaussian kernel
    @param d: the len of the gaussian kernel (kernlen)
    '''
    # Zero padding
    K_p = np.zeros(shape)
    K_p[:d, :d] = K
    # Shift
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)
    # Calcolo FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Moltiplicazione per A
def A(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(np.conj(K) * x))


''' Funzioni per implementazione del metodo del gradiente visto a lezione ( quarto laboratorio )'''

# Procedura di backtracking per la scelta della dimensione del passo
def next_step(x, grad, f_reg):
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad
    j = 0
    jmax = 10
    while ((f_reg(x.reshape(x.size)+alpha*p) > f_reg(x)+c1*alpha*grad.T@p) and j < jmax):
        alpha = rho*alpha
        j += 1
    if (j > jmax):
        return -1
    else:
        return alpha


# Funzione minimize implementata secondo quanto visto a lezione ( quarto laboratorio )
def custom_minimize(x0, b, MAXITERATION, ABSOLUTE_STOP, f_reg, df_reg):
    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))

    # inizializzazione
    x_last = np.copy(x0)

    k = 0

    function_eval_list[k] = f_reg(x_last)
    error_list[k] = np.linalg.norm(x_last - b)
    norm_grad_list[k] = np.linalg.norm(df_reg(x_last))

    while (np.linalg.norm(df_reg(x_last)) > ABSOLUTE_STOP and k < MAXITERATION - 1):
        k = k + 1
        # la direzione è data dal gradiente dell'ultima iterazione
        grad = df_reg(x_last)

        # backtracking step
        step = next_step(x_last, grad, f_reg)


        if (step == -1):
            print('non convergente')
            return (k)  # no convergence

        x_last = x_last - (step * grad).reshape(512, 512)

        function_eval_list[0][k] = f_reg(x_last)
        error_list[0][k] = np.linalg.norm(x_last - b)
        norm_grad_list[0][k] = np.linalg.norm(df_reg(x_last))

    function_eval_list = function_eval_list[0][:k + 1]
    error_list = error_list[0][:k + 1]
    norm_grad_list = norm_grad_list[0][:k + 1]

    return (x_last, norm_grad_list, function_eval_list, error_list, k)


eps = 1e-2


'''Funzioni utili all'implementazione del nuovo termine di regolarizzazione: Variazione Totale '''

# Variazione totale
def totvar(x):
    # Calcola il gradiente di x
    dx, dy = np.gradient(x)
    n2 = np.square(dx) + np.square(dy)

    # Calcola la variazione totale di x
    tv = np.sqrt(n2 + eps ** 2).sum()
    return tv

# Gradiente della variazione totale
def grad_totvar(x):
    # Calcola il numeratore della frazione
    dx, dy = np.gradient(x)

    # Calcola il denominatore della frazione
    n2 = np.square(dx) + np.square(dy)
    den = np.sqrt(n2 + eps ** 2)

    # Calcola le due componenti di F dividendo il gradiente per il denominatore
    Fx = dx / den
    Fy = dy / den

    # Calcola la derivata orizzontale di Fx
    dFdx = np.gradient(Fx, axis=0)

    # Calcola la derivata verticale di Fy
    dFdy = np.gradient(Fy, axis=1)

    # Calcola la divergenza
    div = (dFdx + dFdy)

    # Restituisci il valore del gradiente della variazione totale
    return -div

# Procedura di backtracking per la scelta della dimensione del passo specifica per il nuovo termine di regolarizzazione
def next_step_totvar(x, grad, f_totvar):
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad
    j = 0
    jmax = 10
    while ((f_totvar(x.reshape(x.size) + alpha * p) > f_totvar(x) + c1 * alpha * grad.T@p) and j < jmax):
        alpha = rho * alpha
        j += 1
    if (j > jmax):
        return -1
    else:
        return alpha


# Funzione che implementa il metodo del gradiente come visto a lezione specifica per il nuovo termine di regolarizzazione
def totvar_minimize(x0, b, MAXITERATION, ABSOLUTE_STOP, f_totvar, df_totvar):

    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))

    # Inizializzazione
    x_last = np.copy(x0)

    k = 0

    function_eval_list[k] = f_totvar(x_last)
    error_list[k] = np.linalg.norm(x_last - b)
    norm_grad_list[k] = np.linalg.norm(df_totvar(x_last))

    while (np.linalg.norm(df_totvar(x_last)) > ABSOLUTE_STOP and k < MAXITERATION - 1):
        k = k + 1
        grad = df_totvar(x_last)

        # backtracking step
        step = next_step_totvar(x_last, grad, f_totvar)


        if (step == -1):
            print('non convergente')
            return (k)  # nessuna convergenza

        x_last = x_last - (step * grad).reshape(512, 512)

        function_eval_list[0][k] = f_totvar(x_last)
        error_list[0][k] = np.linalg.norm(x_last - b)
        norm_grad_list[0][k] = np.linalg.norm(df_totvar(x_last))

    function_eval_list = function_eval_list[0][:k + 1]
    error_list = error_list[0][:k + 1]
    norm_grad_list = norm_grad_list[0][:k + 1]

    return (x_last, norm_grad_list, function_eval_list, error_list, k)


# Funzione utile alla comparazione dei diversi valori di PSNR ed MSE nelle immagini ottimizzate con i diversi metodi

def calc_PSNR_MSE(options, img, noised, img_naive, img_reg, img_reg_2, img_totvar, img_name):
    PSNR_noised = metrics.peak_signal_noise_ratio(img, noised)
    MSE_noised = metrics.mean_squared_error(img, noised)

    PSNR_naive = metrics.peak_signal_noise_ratio(img, img_naive)
    MSE_naive = metrics.mean_squared_error(img, img_naive)

    PSNR_reg = metrics.peak_signal_noise_ratio(img, img_reg)
    MSE_reg = metrics.mean_squared_error(img, img_reg)

    PSNR_reg_2 = metrics.peak_signal_noise_ratio(img, img_reg_2)
    MSE_reg_2 = metrics.mean_squared_error(img, img_reg_2)

    PSNR_totvar = metrics.peak_signal_noise_ratio(img, img_totvar)
    MSE_totvar = metrics.mean_squared_error(img, img_totvar)

    # La funzione salva i dati di comparazione in un apposito file ".csv"
    output_PSNR = open(f"sample{img_name}PSNR.csv", 'a')
    output_PSNR.write(
        f"{options},{PSNR_noised},{PSNR_naive},{PSNR_reg},{PSNR_reg_2},{PSNR_totvar}\n")
    output_PSNR.close()
    output_MSE = open(f"sample{img_name}MSE.csv", 'a')
    output_MSE.write(
        f"{options},{MSE_noised},{MSE_naive},{MSE_reg},{MSE_reg_2},{MSE_totvar}\n")
    output_MSE.close()


# Funzione utile a plottare ordinatamente le immagini con l'obiettivo di renderne semplice il confronto
def plot_figure(img, noised, img_naive, img_reg, img_reg_2, img_totvar):
    # ---- PLOTTING ----
    # Immagine originale
    plt.subplot(3, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')
    # Blurred and noised
    plt.subplot(3, 2, 2)
    plt.imshow(noised, cmap='gray')
    plt.title('Blurred and Noised')
    # Correzione Naive
    plt.subplot(3, 2, 3)
    plt.imshow(img_naive, cmap='gray')
    plt.title('Naive correction')
    # Prima regolarizzazione
    plt.subplot(3, 2, 4)
    plt.imshow(img_reg, cmap='gray')
    plt.title('Regolarized correction')
    # Seconda regolarizzazione
    plt.subplot(3, 2, 5)
    plt.imshow(img_reg_2, cmap='gray')
    plt.title('Regolarized 2nd')
    # Correzzione col metodo della variazione totale
    plt.subplot(3, 2, 6)
    plt.imshow(img_totvar, cmap='gray')
    plt.title('TV correction')
    # Stampa finale
    plt.show()
