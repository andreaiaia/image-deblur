import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics, io
from scipy import signal
from scipy.optimize import minimize
from numpy import fft

# Create a Gaussian kernel of size kernlen and standard deviation sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Unidimensional Gaussian kernel
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Bidimensional Gaussian kernel
    kern2d = np.outer(kern1d, kern1d)
    # Normalization
    return kern2d / kern2d.sum()

# Compute the FFT of the kernel 'K' of size 'd' padding with the zeros necessary
# to match the size of 'shape'
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
    # Compute FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Multiplication by A
def A(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(K * x))

# Multiplication by A transpose
def AT(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(np.conj(K) * x))


# ---- PUNTO 2 ----
# Naive deblur function
def f_naive(x):
    X = x.reshape(512, 512)
    res = 0.5 * (np.linalg.norm(A(X, K1) - noised)) ** 2
    return res

def df_naive(x):
    X = x.reshape(512, 512)
    res = AT(A(X, K1) - noised, K1)
    res2 = np.reshape(res, 512 * 512)
    return res2


# ---- PUNTO 3 ----
# Conjugated Gradient method
lambda_value = 0.08

def f_reg(x):
    X = x.reshape(512, 512)
    res = 0.5 * (np.linalg.norm(A(X, K3) - noised)) ** 2 + (lambda_value / 2) * ( np.linalg.norm(noised) ** 2 )
    return res

def df_reg(x):
    X = x.reshape(512, 512)
    res = AT(A(X, K3) - noised, K3) + (lambda_value * noised)
    res2 = np.reshape(res, 512*512)
    return res2

# Gradient method seen during lab lecture (4th lab)

def next_step(x,grad): # backtracking procedure for the choice of the steplength
    alpha=1.1
    rho = 0.5
    c1 = 0.25
    p=-grad
    j=0
    jmax=10
    while ((f_reg(x.reshape(x.size)+alpha*p) > f_reg(x)+c1*alpha*grad.T@p) and j<jmax ):
        alpha= rho*alpha
        j+=1
    if (j>jmax):
        return -1
    else:
        return alpha


def custom_minimize(x0, b, MAXITERATION, ABSOLUTE_STOP):  # funzione che implementa il metodo del gradiente
    # declare x_k and gradient_k vectors

    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))

    # initialize first values
    x_last = np.copy(x0)

    k = 0

    function_eval_list[k] = f_reg(x_last)
    error_list[k] = np.linalg.norm(x_last - b)
    norm_grad_list[k] = np.linalg.norm(df_reg(x_last))

    while (np.linalg.norm(df_reg(x_last)) > ABSOLUTE_STOP and k < MAXITERATION - 1):
        k = k + 1
        grad = df_reg(x_last)  # direction is given by gradient of the last iteration

        # backtracking step
        step = next_step(x_last, grad)
        # Fixed step
        # step = 0.1

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


# ---- TESTS ----
x0 = np.zeros((512,512))

'''
PUNTO 1
Load images and apply blur and noise degradation
'''

# Loading image
img = plt.imread('sample1.png').astype(np.float64)
img = data.camera().astype(np.float64) / 255.0
# Blur filters generation
K1 = psf_fft(gaussian_kernel(5, 0.5), 5, x0.shape)
K2 = psf_fft(gaussian_kernel(7, 1), 7, x0.shape)
K3 = psf_fft(gaussian_kernel(9, 1.3), 9, x0.shape)
# Noise generation
sigma = 0.05
noise = np.random.normal(size = x0.shape) * sigma

# Blurring
blurred = A(img, K3)
# Noising
noised = blurred + noise

'''
PUNTO 2
A first deblur attempt - using the method of Conjugated Gradients with the naive function
'''
res = minimize(f_naive, x0, method='CG', jac=df_naive)
img_naive = res.x.reshape(512, 512)

'''
PUNTO 3
A better deblur attempt
First we use a regolarized Conjugated Gradient method
Then we use another Gradient method seen during the lectures
'''
# Regolarized deblur
res = minimize(f_reg, x0, method='CG', jac=df_reg, options={'maxiter': 100})
img_reg = res.x.reshape(512, 512)
# Second regolarized deblur
(img_reg_2, norm_g_list, fun_eval_list, errors, iterations) = custom_minimize(x0, noised, 50, 1.e-5)

'''
PUNTO 4
'''

eps = 1e-2

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

def f_totvar(x):
    X = x.reshape(512, 512)
    res = 0.5 * (np.linalg.norm(A(X, K3) - noised)) ** 2 + (lambda_value * totvar(noised))
    return res

def df_totvar(x):
    X = x.reshape(512, 512)
    res = AT(A(X, K3) - noised, K3) + (lambda_value * grad_totvar(noised))
    res2 = np.reshape(res, 512*512)
    return res2



def next_step_totvar(x,grad): # backtracking procedure for the choice of the steplength
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


def totvar_minimize(x0, b, MAXITERATION, ABSOLUTE_STOP):  # funzione che implementa il metodo del gradiente
    # declare x_k and gradient_k vectors

    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))

    # initialize first values
    x_last = np.copy(x0)

    k = 0

    function_eval_list[k] = f_totvar(x_last)
    error_list[k] = np.linalg.norm(x_last - b)
    norm_grad_list[k] = np.linalg.norm(df_totvar(x_last))

    while (np.linalg.norm(df_totvar(x_last)) > ABSOLUTE_STOP and k < MAXITERATION - 1):
        k = k + 1
        grad = df_totvar(x_last)  # direction is given by gradient of the last iteration

        # backtracking step
        step = next_step_totvar(x_last, grad)
        # Fixed step
        # step = 0.1

        if (step == -1):
            print('non convergente')
            return (k)  # no convergence

        x_last = x_last - (step * grad).reshape(512, 512)

        function_eval_list[0][k] = f_totvar(x_last)
        error_list[0][k] = np.linalg.norm(x_last - b)
        norm_grad_list[0][k] = np.linalg.norm(df_totvar(x_last))

    function_eval_list = function_eval_list[0][:k + 1]
    error_list = error_list[0][:k + 1]
    norm_grad_list = norm_grad_list[0][:k + 1]


    return (x_last, norm_grad_list, function_eval_list, error_list, k)


(img_totvar, norm_g_list_totvar, fun_eval_list_totvar, errors_totvar, iterations_totvar) = totvar_minimize(x0, noised, 50, 1.e-5)

# ---- PSNR comparison ----
PSNR_noised = metrics.peak_signal_noise_ratio(img, noised)
PSNR_naive = metrics.peak_signal_noise_ratio(img, img_naive)
PSNR_reg = metrics.peak_signal_noise_ratio(img, img_reg)
PSNR_reg_2 = metrics.peak_signal_noise_ratio(img, img_reg_2)
PSNR_totvar = metrics.peak_signal_noise_ratio(img, img_totvar)

print("PSNR comparisons: \n",
    f"\t Noised image: \t\t\t {PSNR_noised} \n",
    f"\t Naive Correction: \t\t {PSNR_naive} \n",
    f"\t Regolarized Correction: \t {PSNR_reg} \n",
    f"\t Regolarized 2nd method: \t {PSNR_reg_2} \n",
    f"\t Regolarized tot_var method: \t {PSNR_totvar} \n")


# ---- PLOTTING ----
# Original image plot
plt.subplot(3,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original img')
# Blurred and noised
plt.subplot(3,2,2)
plt.imshow(noised, cmap='gray')
plt.title('Blurred and Noised')
# Naive correction
plt.subplot(3,2,3)
plt.imshow(img_naive, cmap='gray')
plt.title('Naive correction')
# Regolarized correction
plt.subplot(3,2,4)
plt.imshow(img_reg, cmap='gray')
plt.title('Regolarized correction')
# Second regolarized
plt.subplot(3,2,5)
plt.imshow(img_reg_2, cmap='gray')
plt.title('Regolarized 2nd method')
# tot_var correction
plt.subplot(3,2,6)
plt.imshow(img_totvar, cmap='gray')
plt.title('Tot_var correction')
# It's showtime
plt.show()