console.log('Script running');

// Protocol and Host information ----------------------------

const protocol = window.location.protocol;
const host = window.location.host;

let baseURL;
// let baseStreamURL;
const AWSServerIP = "15.229.156.138";
const AWSServerURL = `http://${AWSServerIP}`;
const cloudRunServerURL = 'https://video-analytics-oayt5ztuxq-ue.a.run.app'; // cloud run server

// Check if the host is a server 
// Check if the host is hostgator or the local file system
if (host.includes("octacity.org") | host == "") {
    baseURL = AWSServerURL;
    // baseStreamURL = AWSServerURL;
}
else {
  baseURL = ".";
  // baseStreamURL = ".";
}

// Log the final base URL and host
console.log('Host:', host);
console.log('Base URL:', baseURL);
// console.log('Base Streaming URL:', baseStreamURL);

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM content loaded');
    // const baseURL = "http://127.0.0.1:5000";
    const camerasPerPage = 10; // Número de câmeras a serem exibidas por página

    let camerasData = []; // Array para armazenar os dados das câmeras obtidos
    let currentPage = 1; // Número da página atual para paginação
    let filteredCameras = []; // Array para armazenar as câmeras filtradas

    const camerasContainer = document.getElementById('camerasContainer');
    const paginationContainer = document.getElementById('paginationContainer');
    const createCameraForm = document.getElementById('createCameraForm');
    const closeModalBtns = document.querySelectorAll('.modal #closeModalBtn');
    const cancelCreateBtn = document.getElementById('cancelCreateBtn');
    const errorContainer = document.getElementById('errorContainer');

    // Obter dados das câmeras da API
    fetch(`${baseURL}/cameras`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Falha ao obter as câmeras');
            }
            return response.json();
        })
        .then(data => {
            camerasData = data;
            filteredCameras = camerasData;
            renderCameras();
            renderPagination();
        })
        .catch(error => {
            console.error(error);
            showError('Falha ao obter as câmeras. Por favor, tente novamente.');
        });

    // Renderizar lista de câmeras
    function renderCameras() {
        camerasContainer.innerHTML = '';

        const start = (currentPage - 1) * camerasPerPage;
        const end = start + camerasPerPage;
        const camerasToShow = filteredCameras.slice(start, end);

        camerasToShow.forEach(camera => {
            const cameraItem = document.createElement('div');
            cameraItem.className = 'camera-item';
            cameraItem.dataset.id = camera.id;
            cameraItem.addEventListener('click', () => {
                openModal(camera);
            });

            // Adicionar detalhes da câmera
            const cameraName = document.createElement('h3');
            cameraName.textContent = camera.name;
            cameraItem.appendChild(cameraName);

            const cameraUrl = document.createElement('p');
            cameraUrl.textContent = 'URL: ' + camera.url;
            cameraItem.appendChild(cameraUrl);

            const cameraObjects = document.createElement('p');
            cameraObjects.textContent = 'Objetos: ' + camera.objects;
            cameraItem.appendChild(cameraObjects);

            const cameraPostUrl = document.createElement('p');
            cameraPostUrl.textContent = 'Post URL: ' + camera.post_url;
            cameraItem.appendChild(cameraPostUrl);

            // Adicionar botões de edição e exclusão
            const editButton = document.createElement('button');
            editButton.textContent = 'Editar';
            editButton.addEventListener('click', (event) => {
                event.stopPropagation();
                // Confirmar a ação de edição
                showConfirmation('Tem certeza de que deseja prosseguir com a edição?', () => {
                    editCamera(camera.id);
                });
            });
            cameraItem.appendChild(editButton);

            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Excluir';
            deleteButton.addEventListener('click', (event) => {
                event.stopPropagation();
                // Confirmar a ação de exclusão
                showConfirmation('Tem certeza de que deseja excluir esta câmera?', () => {
                    deleteCamera(camera.id);
                });
            });
            cameraItem.appendChild(deleteButton);

            // Adicionar item da câmera ao container
            camerasContainer.appendChild(cameraItem);
        });
    }

    // Renderizar paginação
    function renderPagination() {
        const totalPages = getTotalPages();

        if (totalPages > 1) {
            const prevButton = createPaginationButton('Anterior', currentPage > 1, previousPage);
            paginationContainer.appendChild(prevButton);

            const goToPageInput = document.createElement('input');
            goToPageInput.type = 'number';
            goToPageInput.min = 1;
            goToPageInput.max = totalPages;
            goToPageInput.value = currentPage;
            goToPageInput.addEventListener('input', () => {
                const page = parseInt(goToPageInput.value);
                if (page >= 1 && page <= totalPages) {
                    goToPage(page);
                }
            });
            paginationContainer.appendChild(goToPageInput);

            const nextButton = createPaginationButton('Próxima', currentPage < totalPages, nextPage);
            paginationContainer.appendChild(nextButton);

            const firstButton = createPaginationButton('Primeira', currentPage > 1, () => goToPage(1));
            paginationContainer.prepend(firstButton);

            const lastButton = createPaginationButton('Última', currentPage < totalPages, () => goToPage(totalPages));
            paginationContainer.appendChild(lastButton);
        }
    }

    // Criar botão de paginação
    function createPaginationButton(label, enabled, onClick) {
        const button = document.createElement('button');
        button.textContent = label;
        button.disabled = !enabled;
        button.addEventListener('click', onClick);
        return button;
    }

    // Ir para uma página específica
    function goToPage(page) {
        currentPage = page;
        renderCameras();
        renderPagination();
    }

    // Ir para a página anterior
    function previousPage() {
        if (currentPage > 1) {
            currentPage--;
            renderCameras();
            renderPagination();
        }
    }

    // Ir para a próxima página
    function nextPage() {
        const totalPages = getTotalPages();
        if (currentPage < totalPages) {
            currentPage++;
            renderCameras();
            renderPagination();
        }
    }

    // Obter o número total de páginas
    function getTotalPages() {
        return Math.ceil(filteredCameras.length / camerasPerPage);
    }

    // Pesquisar câmeras
    function searchCameras() {
        const searchInput = document.getElementById('searchInput').value.toLowerCase();
        filteredCameras = camerasData.filter(camera => (camera.name && camera.name.toLowerCase().includes(searchInput)) || (camera.url) && camera.url.toLowerCase().includes(searchInput));
        currentPage = 1;
        renderCameras();
        renderPagination();
    }

    // Criar câmera
    function createCamera(event) {
        event.preventDefault();

        const cameraNameInput = document.getElementById('cameraNameInput');
        const cameraUrlInput = document.getElementById('cameraUrlInput');
        const cameraObjectsInput = document.getElementById('cameraObjectsInput');
        const cameraPostUrlInput = document.getElementById('cameraPostUrlInput');

        const newCamera = {
            name: cameraNameInput.value,
            url: cameraUrlInput.value,
            objects: cameraObjectsInput.value,
            post_url: cameraPostUrlInput.value
        };

        // Chamar o endpoint da API para criar a câmera usando os dados fornecidos
        fetch(`${baseURL}/cameras`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newCamera)
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error('Falha ao criar a câmera');
                }
            })
            .then(camera => {
                // Adicionar a nova câmera aos arrays de dados
                camerasData.push(camera);
                filteredCameras.push(camera);

                // Limpar o formulário de criação
                cameraNameInput.value = '';
                cameraUrlInput.value = '';
                cameraObjectsInput.value = '';
                cameraPostUrlInput.value = '';

                // Fechar o modal de criação e atualizar a interface
                closeModal();
                renderCameras();
                renderPagination();
            })
            .catch(error => {
                console.error(error);
                showError('Falha ao criar a câmera. Por favor, tente novamente.');
            });
    }

    // Editar câmera
    function editCamera(cameraId) {
        // Redirecionar para a tela de edição da câmera ou implementar a edição inline
        // com base no design e requisitos da sua plataforma
    }

    // Excluir câmera
    function deleteCamera(cameraId) {
        const camera = filteredCameras.find(camera => camera.id === cameraId);

        if (camera) {
            // Confirmar a exclusão com uma caixa de diálogo modal
            showConfirmation('Tem certeza de que deseja excluir esta câmera?', () => {
                // Chamar o endpoint da API para excluir a câmera usando sua URL
                fetch(`${baseURL}/cameras`, { method: 'DELETE' })
                    .then(response => {
                        if (response.ok) {
                            // Remover a câmera dos arrays de dados
                            camerasData = camerasData.filter(camera => camera.id !== cameraId);
                            filteredCameras = filteredCameras.filter(camera => camera.id !== cameraId);

                            // Atualizar a interface
                            renderCameras();
                            renderPagination();
                        } else {
                            throw new Error('Falha ao excluir a câmera');
                        }
                    })
                    .catch(error => {
                        console.error(error);
                        showError('Falha ao excluir a câmera. Por favor, tente novamente.');
                    });
            });
        }
    }

    // Abrir modal de criação de câmera
    function openCreateModal() {
        document.getElementById('createCameraModal').style.display = 'block';
    }

    // Fechar modal de criação de câmera
    function closeCreateModal() {
        document.getElementById('createCameraModal').style.display = 'none';
        clearError();
    }

    // Abrir modal de visualização/edição de câmera
    function openModal(camera) {
        // Implemente o código para abrir o modal e exibir os detalhes da câmera
    }

    // Fechar modal de visualização/edição de câmera
    function closeModal() {
        // Implemente o código para fechar o modal
    }

    // Exibir uma caixa de diálogo modal de confirmação
    function showConfirmation(message, onConfirm) {
        // Implemente o código para exibir uma caixa de diálogo modal de confirmação
    }

    // Exibir uma mensagem de erro
    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
    }

    // Limpar a mensagem de erro
    function clearError() {
        errorContainer.textContent = '';
        errorContainer.style.display = 'none';
    }

    // Event listeners
    document.getElementById('searchInput').addEventListener('input', searchCameras);
    document.getElementById('createCameraBtn').addEventListener('click', openCreateModal);
    createCameraForm.addEventListener('submit', createCamera);
    closeModalBtns.forEach(btn => btn.addEventListener('click', closeCreateModal));
    cancelCreateBtn.addEventListener('click', closeCreateModal);
});
