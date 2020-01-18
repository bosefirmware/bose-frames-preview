/**
 * @namespace
 */
/* istanbul ignore next */
var Bose = Bose || {};

/**
 * @namespace
 */
/* istanbul ignore next */
Bose.AllowedStatesInfo =  Bose.BaseClass.extend(function() {
    var selectors = {
        content  : '.js-content',
        button   : '.js-show-button',
        icon     : '.js-icon'
    };

    /**
     * Bind events.
     */
    function bindEvents() {
        var _this = this;

        this.$el.on('click', selectors.button, function(event) {
            event.preventDefault();
            var $button = $jq(this).find('.js-button-text');

            if (_this.$content.hasClass('bose-allowedStatesInfo__content--active')) {
                _this.$content.removeClass('bose-allowedStatesInfo__content--active');
                _this.$icon.removeClass('-minus');
                $button.text(_this.showmorelabel);
            } else {
                _this.$content.addClass('bose-allowedStatesInfo__content--active');
                _this.$icon.addClass('-minus');
                $button.text(_this.showlesslabel);
            }
        });
    }

    return {
        /**
         * Jquery refernce to content element.
         */
        $content: null,
        /**
         * Jquery refernce to icon.
         */
        $icon: null,
        /**
         * Initializes the AllowedStatesInfo component.
         */
        _setup : function _setup() {
            this._super();
            this.$content = $jq(selectors.content, this.$el);
            this.$icon = $jq(selectors.icon, this.$el);
            bindEvents.call(this);
        }
    };
}());
$jq(document).ready(function(){var a=function a(){$jq('[data-mod="allowedstatesinfo"]:not(.initialized)').each(function(){new Bose.AllowedStatesInfo({el:this})})};a(),$jq(window).on("shown.popup",a)});
