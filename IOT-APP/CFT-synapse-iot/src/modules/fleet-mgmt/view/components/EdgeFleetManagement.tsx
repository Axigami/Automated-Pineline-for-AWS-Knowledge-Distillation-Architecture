import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Plus,
  Cpu,
  Database,
  Wifi,
  Map as MapIcon,
  MoreVertical,
  RefreshCw,
  FileText,
  Thermometer,
  Activity,
  Server,
  MapPin,
  X
} from 'lucide-react';
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  YAxis
} from 'recharts';
import { motion } from 'motion/react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import type { EdgeNodeUIModel, TelemetryPoint } from '../../model/types';
import { NodeCard, NodeCardSkeleton } from './NodeCard';
import { AddNodeModal } from './AddNodeModal';
import { useLanguage } from '../../../../core/i18n/LanguageContext';

// ─── Tọa độ giả lập tham khảo cho các tỉnh (GPS) ───
const LOCATION_COORDS: Record<string, [number, number]> = {
  // --- Miền Bắc ---
  'hà nội': [21.0285, 105.8542], 'hanoi': [21.0285, 105.8542], 'ha noi': [21.0285, 105.8542],
  'hải phòng': [20.8449, 106.6881], 'hai phong': [20.8449, 106.6881],
  'hà giang': [22.8233, 104.9836], 'ha giang': [22.8233, 104.9836],
  'cao bằng': [22.6667, 106.2500], 'cao bang': [22.6667, 106.2500],
  'bắc kạn': [22.1470, 105.8348], 'bac kan': [22.1470, 105.8348],
  'tuyên quang': [21.8229, 105.2152], 'tuyen quang': [21.8229, 105.2152],
  'lào cai': [22.4836, 103.9667], 'lao cai': [22.4836, 103.9667],
  'điện biên': [21.3853, 103.0232], 'dien bien': [21.3853, 103.0232],
  'lai châu': [22.3969, 103.4617], 'lai chau': [22.3969, 103.4617],
  'sơn la': [21.3280, 103.9149], 'son la': [21.3280, 103.9149],
  'yên bái': [21.7229, 104.9113], 'yen bai': [21.7229, 104.9113],
  'hoà bình': [20.8133, 105.3384], 'hoa binh': [20.8133, 105.3384],
  'thái nguyên': [21.5928, 105.8443], 'thai nguyen': [21.5928, 105.8443],
  'lạng sơn': [21.8333, 106.7667], 'lang son': [21.8333, 106.7667],
  'quảng ninh': [21.0069, 107.2925], 'quang ninh': [21.0069, 107.2925],
  'bắc giang': [21.2731, 106.1946], 'bac giang': [21.2731, 106.1946],
  'phú thọ': [21.3167, 105.2167], 'phu tho': [21.3167, 105.2167],
  'vĩnh phúc': [21.3083, 105.5977], 'vinh phuc': [21.3083, 105.5977],
  'bắc ninh': [21.1861, 106.0763], 'bac ninh': [21.1861, 106.0763],
  'hải dương': [20.9381, 106.3153], 'hai duong': [20.9381, 106.3153],
  'hưng yên': [20.8500, 106.0500], 'hung yen': [20.8500, 106.0500],
  'thái bình': [20.4468, 106.3409], 'thai binh': [20.4468, 106.3409],
  'hà nam': [20.5408, 105.9122], 'ha nam': [20.5408, 105.9122],
  'nam định': [20.4333, 106.1667], 'nam dinh': [20.4333, 106.1667],
  'ninh bình': [20.2539, 105.9753], 'ninh binh': [20.2539, 105.9753],

  // --- Miền Trung ---
  'thanh hóa': [19.8075, 105.7765], 'thanh hoa': [19.8075, 105.7765],
  'nghệ an': [19.3892, 104.8973], 'nghe an': [19.3892, 104.8973], 'vinh': [18.6667, 105.6667],
  'hà tĩnh': [18.3428, 105.9056], 'ha tinh': [18.3428, 105.9056],
  'quảng bình': [17.4833, 106.6000], 'quang binh': [17.4833, 106.6000], 'đồng hới': [17.4833, 106.6000],
  'quảng trị': [16.7497, 107.1856], 'quang tri': [16.7497, 107.1856],
  'thừa thiên huế': [16.4637, 107.5905], 'thua thien hue': [16.4637, 107.5905], 'huế': [16.4637, 107.5905], 'hue': [16.4637, 107.5905],
  'đà nẵng': [16.0471, 108.2068], 'da nang': [16.0471, 108.2068],
  'quảng nam': [15.5674, 107.9945], 'quang nam': [15.5674, 107.9945], 'tam kỳ': [15.5674, 108.4682], 'hội an': [15.8794, 108.3350],
  'quảng ngãi': [15.1205, 108.7923], 'quang ngai': [15.1205, 108.7923],
  'bình định': [14.1611, 108.9050], 'binh dinh': [14.1611, 108.9050], 'quy nhơn': [13.7758, 109.2189],
  'phú yên': [13.0883, 109.3211], 'phu yen': [13.0883, 109.3211], 'tuy hòa': [13.0883, 109.3211],
  'khánh hòa': [12.2450, 109.1947], 'khanh hoa': [12.2450, 109.1947], 'nha trang': [12.2450, 109.1943],
  'ninh thuận': [11.5833, 108.9833], 'ninh thuan': [11.5833, 108.9833], 'phan rang': [11.5833, 108.9833],
  'bình thuận': [10.9333, 108.1000], 'binh thuan': [10.9333, 108.1000], 'phan thiết': [10.9259, 108.1060],

  // --- Tây Nguyên ---
  'kon tum': [14.3500, 108.0000],
  'gia lai': [13.9833, 108.0000], 'pleiku': [13.9833, 108.0000],
  'đắk lắk': [12.6667, 108.0444], 'dak lak': [12.6667, 108.0444], 'buôn ma thuột': [12.6667, 108.0444],
  'đắk nông': [12.0019, 107.6975], 'dak nong': [12.0019, 107.6975],
  'lâm đồng': [11.9404, 108.4583], 'lam dong': [11.9404, 108.4583], 'đà lạt': [11.9404, 108.4583], 'da lat': [11.9404, 108.4583],

  // --- Miền Nam ---
  'bình phước': [11.7511, 106.9157], 'binh phuoc': [11.7511, 106.9157],
  'tây ninh': [11.3000, 106.1000], 'tay ninh': [11.3000, 106.1000],
  'bình dương': [11.0000, 106.6667], 'binh duong': [11.0000, 106.6667],
  'đồng nai': [10.9575, 106.8427], 'dong nai': [10.9575, 106.8427], 'biên hòa': [10.9575, 106.8427],
  'bà rịa': [10.4960, 107.1685], 'ba ria': [10.4960, 107.1685], 'vũng tàu': [10.3459, 107.0842], 'vung tau': [10.3459, 107.0842],
  'hồ chí minh': [10.8231, 106.6297], 'ho chi minh': [10.8231, 106.6297], 'hcm': [10.8231, 106.6297], 'hcmc': [10.8231, 106.6297], 'sài gòn': [10.8231, 106.6297], 'sai gon': [10.8231, 106.6297],

  // --- Miền Tây ---
  'long an': [10.5333, 106.4000], 'tân an': [10.5333, 106.4000],
  'tiền giang': [10.3601, 106.3473], 'tien giang': [10.3601, 106.3473], 'mỹ tho': [10.3601, 106.3473],
  'bến tre': [10.2333, 106.3667], 'ben tre': [10.2333, 106.3667],
  'trà vinh': [9.9333, 106.3500], 'tra vinh': [9.9333, 106.3500],
  'vĩnh long': [10.2500, 105.9667], 'vinh long': [10.2500, 105.9667],
  'đồng tháp': [10.4578, 105.6267], 'dong thap': [10.4578, 105.6267], 'cao lãnh': [10.4578, 105.6267],
  'an giang': [10.3833, 105.4167], 'long xuyên': [10.3833, 105.4167], 'châu đốc': [10.7000, 105.1167],
  'kiên giang': [10.0167, 105.0833], 'kien giang': [10.0167, 105.0833], 'rạch giá': [10.0167, 105.0833], 'phú quốc': [10.2289, 103.9573],
  'cần thơ': [10.0452, 105.7469], 'can tho': [10.0452, 105.7469],
  'hậu giang': [9.7833, 105.4667], 'hau giang': [9.7833, 105.4667], 'vị thanh': [9.7833, 105.4667],
  'sóc trăng': [9.6000, 105.9833], 'soc trang': [9.6000, 105.9833],
  'bạc liêu': [9.2833, 105.7167], 'bac lieu': [9.2833, 105.7167],
  'cà mau': [9.1833, 105.1500], 'ca mau': [9.1833, 105.1500],

  // --- Biển Đảo ---
  'hoàng sa': [16.5000, 112.0000], 'hoang sa': [16.5000, 112.0000], 'quần đảo hoàng sa': [16.5000, 112.0000],
  'trường sa': [8.6333, 111.9167], 'truong sa': [8.6333, 111.9167], 'quần đảo trường sa': [8.6333, 111.9167]
};

import L from 'leaflet';

// Helper function to extract base coordinates from known locations
function getBaseCoords(locationStr: string): [number, number] | null {
  const loc = locationStr.toLowerCase();
  for (const key of Object.keys(LOCATION_COORDS)) {
    if (loc.includes(key)) return LOCATION_COORDS[key];
  }
  return null;
}

// Define the custom pulsing icon factory
const createPulseIcon = (status: 'online' | 'offline') => {
  const bgClass = status === 'online' ? 'bg-emerald-500' : 'bg-red-500';
  const shadowClass =
    status === 'online'
      ? 'shadow-[0_0_8px_rgba(16,185,129,0.8)]'
      : 'shadow-[0_0_8px_rgba(239,68,68,0.8)]';
  const pulseClass = status === 'online' ? 'bg-emerald-500/30' : 'bg-red-500/30';

  const html = `
    <div class="relative flex items-center justify-center w-full h-full">
      <div class="absolute w-2 h-2 rounded-full ${bgClass} ${shadowClass}"></div>
      <div class="absolute w-6 h-6 rounded-full ${pulseClass} animate-ping opacity-75"></div>
    </div>
  `;

  return L.divIcon({
    html,
    className: 'custom-pulse-marker',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -12]
  });
};

interface EdgeFleetManagementProps {
  nodes: EdgeNodeUIModel[];
  telemetryMap?: Record<string, TelemetryPoint[]>;
  isLoading?: boolean;
  error?: string | null;
  selectedNodeId?: string | null;
  onSelectNode?: (id: string | null) => void;
  onRefresh?: () => Promise<void>;
  onRestart?: (id: string) => Promise<void>;
  onDelete?: (id: string) => Promise<void>;
  onEditLocation?: (id: string, newLocation: string) => Promise<void>;
  onAddNode?: (payload: any) => Promise<void>;
}

// ─── Component điều khiển di chuyển bản đồ ───
const MapFlyController: React.FC<{ focusedNodeId: string | null, markerRefs: React.MutableRefObject<Record<string, any>> }> = ({ focusedNodeId, markerRefs }) => {
  const map = useMap();
  const isInitialMount = React.useRef(true);

  React.useEffect(() => {
    // Bỏ qua animate khi component mới render lần đầu
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    if (focusedNodeId && markerRefs.current[focusedNodeId]) {
      const marker = markerRefs.current[focusedNodeId];
      // 1. Bay tới tọa độ của marker
      map.flyTo(marker.getLatLng(), 10, { animate: true, duration: 1.5 });
      // 2. Tự động mở thẻ Popup hiển thị thông tin
      marker.openPopup();
    } else if (focusedNodeId === null) {
      // 3. Trả về màn hình toàn cảnh Việt Nam khi đóng thẻ
      map.flyToBounds([
        [8.4, 102.1],
        [23.4, 109.5]
      ], { animate: true, duration: 1.5 });
    }
  }, [focusedNodeId, map, markerRefs]);

  return null;
};

// ─── Component sửa lỗi Flexbox làm Map bị hụt gạch (Tiles) ───
const MapResizer: React.FC = () => {
  const map = useMap();
  React.useEffect(() => {
    let isMounted = true;
    const observer = new ResizeObserver(() => {
      if (isMounted) map.invalidateSize();
    });
    observer.observe(map.getContainer());

    // Fallback trigger sau khi component mount (đề phòng React nháy DOM)
    const timeout = setTimeout(() => {
      if (isMounted) map.invalidateSize();
    }, 200);

    return () => {
      isMounted = false;
      clearTimeout(timeout);
      observer.disconnect();
    };
  }, [map]);
  return null;
};

export const EdgeFleetManagement: React.FC<EdgeFleetManagementProps> = ({
  nodes,
  telemetryMap = {},
  isLoading = false,
  error = null,
  onRestart,
  onDelete,
  onEditLocation,
  onAddNode
}) => {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const { t } = useLanguage();
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);

  // Search Highlight Logic
  const [searchParams, setSearchParams] = useSearchParams();
  const [bannerNodeId, setBannerNodeId] = useState<string | null>(null);
  const nodeIdFromSearch = searchParams.get('nodeId');

  useEffect(() => {
    if (!nodeIdFromSearch) return;
    setBannerNodeId(nodeIdFromSearch);
    setSearchParams(
      (prev) => {
        const n = new URLSearchParams(prev);
        n.delete('nodeId');
        return n;
      },
      { replace: true },
    );
  }, [nodeIdFromSearch, setSearchParams]);

  // Tham chiếu tới các thẻ Marker trên bản đồ để đóng/mở Popup bằng code
  const markerRefs = React.useRef<Record<string, any>>({});

  // Tiền xử lý tính toán tọa độ (kèm Jittering mấp mô 5km) cho TẤT CẢ device trước khi render
  const mappedNodes = React.useMemo(() => {
    const locationCounts: Record<string, number> = {};
    return nodes.map((node, i) => {
      let baseCoords = getBaseCoords(node.location);
      if (!baseCoords) {
        baseCoords = [16.0 + (i % 5 - 2) * 1.5, 108.0 + (i % 3 - 1) * 1.5];
      }

      const locKey = `${baseCoords[0]},${baseCoords[1]}`;
      const count = locationCounts[locKey] || 0;
      locationCounts[locKey] = count + 1;

      let finalCoords = [...baseCoords] as [number, number];
      if (count > 0) {
        const radius = 0.05 + (Math.floor(count / 6) * 0.05);
        const angle = (count * Math.PI * 2) / 6;
        finalCoords = [
          baseCoords[0] + radius * Math.sin(angle),
          baseCoords[1] + radius * Math.cos(angle)
        ];
      }
      return { ...node, computedCoords: finalCoords };
    });
  }, [nodes]);

  const onlineCount = nodes.filter(n => n.status === 'online').length;
  const offlineCount = nodes.filter(n => n.status !== 'online').length;
  const totalNodes = nodes.length;
  const healthPct = totalNodes > 0 ? (onlineCount / totalNodes) * 100 : 0;
  return (
    <div className="space-y-6">
      {bannerNodeId && (
        <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-200">
          <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400/90">Opened from search</span>
          <p className="mt-1 font-mono text-xs text-slate-300 break-all">{bannerNodeId}</p>
          <p className="mt-1 text-xs text-slate-500">Fleet cards below are demo data; connect real node list to highlight this id.</p>
        </div>
      )}

      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">{t('edgeFleet', 'title') || 'Edge Node Fleet'}</h2>
          <p className="text-slate-400 text-sm">{t('edgeFleet', 'subTitle') || 'Managing hardware layer and Raspberry Pi nodes'}</p>
        </div>
        <button
          onClick={() => setIsAddModalOpen(true)}
          className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-bold transition-all shadow-lg shadow-blue-600/20 flex items-center gap-2"
        >
          <Plus size={18} /> {t('edgeFleet', 'addNode') || 'Add New Node'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Fleet List */}
        <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
          {isLoading && nodes.length === 0 ? (
            // Show skeletons on initial load
            [...Array(4)].map((_, i) => <NodeCardSkeleton key={i} />)
          ) : error ? (
            <div className="col-span-full bg-red-900/20 border border-red-500/50 p-4 rounded-xl text-red-400 text-sm flex items-center gap-2">
              <Activity size={16} /> {error}
            </div>
          ) : (
            mappedNodes.map(node => (
              <NodeCard
                key={node.id}
                node={node}
                telemetry={telemetryMap[node.id] || []}
                onRestart={onRestart}
                onDelete={onDelete}
                onEditLocation={onEditLocation}
                onClick={() => setFocusedNodeId(node.id)}
              />
            ))
          )}
        </div>

        {/* Fleet Map & Stats */}
        <div className="flex flex-col gap-6 h-full">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col min-h-[300px]">
            <div className="flex items-center gap-2 mb-4 shrink-0">
              <MapIcon size={18} className="text-blue-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">{t('edgeFleet', 'fleetMap') || 'Fleet Map'}</h3>
            </div>
            <div className="flex-1 w-full bg-slate-950 rounded-lg border border-slate-800 relative overflow-hidden group z-0">
              {/* Leaflet JS Intergration */}
              <MapContainer
                bounds={[
                  [8.4, 102.1], // Cà Mau / Phú Quốc (Góc Tây Nam)
                  [23.4, 109.5] // Hà Giang / Biển Đông (Góc Đông Bắc)
                ]}
                minZoom={4}
                maxZoom={10}
                maxBounds={[
                  [0.0, 90.0], // Nam/Tây
                  [35.0, 125.0] // Bắc/Đông - Mở rộng cực đại để Popup mập mạp không bị kẹt trần
                ]}
                maxBoundsViscosity={0.8}
                scrollWheelZoom={true}
                className="w-full h-full"
                style={{ background: '#020617' }} // Explicitly override Leaflet's default #ddd gray void
                zoomControl={false}
                attributionControl={false}
              >
                <MapResizer />
                {/* Plugin bắt sự kiện FlyTo và Tự động mở Popup */}
                <MapFlyController focusedNodeId={focusedNodeId} markerRefs={markerRefs} />

                {/* CartoDB Dark Matter (Perfect for Dark Theme) */}
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {/* Map Context Render */}
                {mappedNodes.map((node) => (
                  <Marker
                    key={node.id}
                    position={node.computedCoords}
                    icon={createPulseIcon(node.status === 'offline' ? 'offline' : 'online')}
                    ref={(r) => {
                      if (r) markerRefs.current[node.id] = r;
                    }}
                    eventHandlers={{
                      popupclose: () => setFocusedNodeId(null)
                    }}
                  >
                    <Popup className="custom-popup" closeButton={false}>
                      <div className="flex flex-col gap-3 min-w-[160px] p-1 font-sans relative">
                        {/* Custom Close Button Without href */}
                        <button
                          onClick={() => {
                            markerRefs.current[node.id]?.closePopup();
                          }}
                          className="absolute -top-2 -right-2 p-1 text-slate-500 hover:text-red-400 hover:bg-slate-800/80 rounded-full transition-colors z-10"
                        >
                          <X size={14} />
                        </button>

                        {/* Header */}
                        <div className="flex items-center gap-2 border-b border-slate-700/50 pb-2">
                          <Server size={14} className="text-blue-400" />
                          <strong className="text-sm font-bold text-slate-100">{node.nodeCode}</strong>
                        </div>

                        {/* Info */}
                        <div className="flex justify-between items-center text-xs">
                          <div className="flex items-center gap-1 text-slate-400">
                            <MapPin size={10} /> {node.location}
                          </div>
                          <div className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider
                                ${node.status === 'online' ? 'bg-emerald-500/20 text-emerald-400' :
                              node.status === 'offline' ? 'bg-red-500/20 text-red-500' :
                                'bg-amber-500/20 text-amber-500'}
                              `}>
                            {node.status}
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="grid grid-cols-2 gap-2 mt-1">
                          <div className="bg-slate-800/80 rounded p-1.5 text-center shadow-inner">
                            <span className="block text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-0.5">CPU</span>
                            <span className="text-xs font-bold text-slate-200">{node.cpuPct}</span>
                          </div>
                          <div className="bg-slate-800/80 rounded p-1.5 text-center shadow-inner">
                            <span className="block text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-0.5">RAM</span>
                            <span className="text-xs font-bold text-slate-200">{node.memPct}</span>
                          </div>
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                ))
                }
              </MapContainer>

              <div className="absolute bottom-3 left-3 bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded px-2 py-1 text-[10px] font-bold text-slate-400 pointer-events-none z-[1000]">
                Map: CartoDB Dark Matter
              </div>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl shrink-0">
            <div className="flex items-center gap-2 mb-4">
              <Activity size={18} className="text-blue-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">{t('edgeFleet', 'healthStatus') || 'Fleet Health'}</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">{t('edgeFleet', 'totalNodes') || 'Total Nodes'}</span>
                <span className="text-sm font-bold text-slate-200">{totalNodes}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">{t('edgeFleet', 'online') || 'Online'}</span>
                <span className="text-sm font-bold text-emerald-400">{onlineCount}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">{t('edgeFleet', 'offline') || 'Offline'}</span>
                <span className="text-sm font-bold text-rose-500">{offlineCount}</span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${healthPct}%` }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <AddNodeModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSubmit={async (payload) => {
          if (onAddNode) {
            await onAddNode(payload);
          }
        }}
      />
    </div>
  );
};
