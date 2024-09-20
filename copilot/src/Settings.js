import React, { useState, useEffect } from 'react';
import { TextField, Button, Select, MenuItem, Chip, Box, FormControl, InputLabel } from '@mui/material';
import axios from 'axios';

function Settings({ onClearChat }) {
    const [settings, setSettings] = useState({
        default_portfolio_value: 100000,
        risk_profile: 'moderate',
        preferred_sectors: [],
        theme: 'light'
    });

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const response = await axios.get(`${process.env.REACT_APP_API_URL}/settings`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            setSettings(response.data);
        } catch (error) {
            console.error("Erreur lors de la récupération des paramètres:", error);
            if (error.response) {
                console.error("Données de réponse:", error.response.data);
            }
            alert(`Erreur lors de la récupération des paramètres: ${error.message}`);
        }
    };

    const handleSettingsChange = (setting, value) => {
        setSettings(prevSettings => ({ ...prevSettings, [setting]: value }));
    };

    const clearChat = async () => {
        try {
            const token = localStorage.getItem('token');
            await axios.post(process.env.REACT_APP_API_URL + '/clear_chat', {}, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            alert('Historique du chat effacé avec succès');
            onClearChat();
        } catch (error) {
            console.error('Erreur lors de l\'effacement de l\'historique du chat:', error);
            alert('Erreur lors de l\'effacement de l\'historique du chat. Veuillez réessayer.');
        }
    };

    const saveSettings = async () => {
        try {
            const response = await axios.post(
                `${process.env.REACT_APP_API_URL}/settings`,
                settings,  // Envoyer les données directement
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                }
            );

            console.log("Réponse du serveur:", response);

            if (response.status === 200) {
                alert('Paramètres sauvegardés avec succès');
            } else {
                throw new Error(`Erreur ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error("Erreur détaillée:", error);
            if (error.response) {
                console.error("Données de réponse:", error.response.data);
            }
            alert(`Erreur lors de la sauvegarde des paramètres: ${error.message}`);
        }
    };

    return (
        <Box>
            <TextField
                label="Valeur par défaut du portefeuille"
                value={settings.default_portfolio_value}
                onChange={(e) => handleSettingsChange('default_portfolio_value', e.target.value)}
                type="number"
                fullWidth
                margin="normal"
            />
            <FormControl fullWidth margin="normal">
                <InputLabel>Profil de risque</InputLabel>
                <Select
                    value={settings.risk_profile}
                    onChange={(e) => handleSettingsChange('risk_profile', e.target.value)}
                >
                    <MenuItem value="low">Faible</MenuItem>
                    <MenuItem value="moderate">Modéré</MenuItem>
                    <MenuItem value="high">Élevé</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Secteurs préférés</InputLabel>
                <Select
                    multiple
                    value={settings.preferred_sectors}
                    onChange={(e) => handleSettingsChange('preferred_sectors', e.target.value)}
                    renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {selected.map((value) => (
                                <Chip key={value} label={value} />
                            ))}
                        </Box>
                    )}
                >
                    <MenuItem value="Technology">Technologie</MenuItem>
                    <MenuItem value="Healthcare">Santé</MenuItem>
                    <MenuItem value="Finance">Finance</MenuItem>
                    <MenuItem value="Energy">Énergie</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Thème</InputLabel>
                <Select
                    value={settings.theme}
                    onChange={(e) => handleSettingsChange('theme', e.target.value)}
                >
                    <MenuItem value="light">Clair</MenuItem>
                    <MenuItem value="dark">Sombre</MenuItem>
                </Select>
            </FormControl>
            <Button
                onClick={clearChat}
                variant="contained"
                color="secondary"
                style={{ marginTop: '20px', marginRight: '10px' }}
            >
                Effacer l'historique du chat
            </Button>
            <Button 
                onClick={saveSettings} 
                variant="contained" 
                color="primary"
                style={{ marginTop: '20px' }}
            >
                Sauvegarder les paramètres
            </Button>
        </Box>
    );
}

export default Settings;