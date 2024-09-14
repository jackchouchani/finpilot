import axios from 'axios';

export const login = async (username, password) => {
    try {
        const response = await axios.post('${process.env.REACT_APP_API_URL}/login', { username, password });
        const { access_token } = response.data;
        localStorage.setItem('token', access_token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        return true;
    } catch (error) {
        console.error("Login failed", error);
        return false;
    }
};

export const logout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
};